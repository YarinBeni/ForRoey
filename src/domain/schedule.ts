/**
 * Slot scheduling: turns a day's target count into concrete, randomised but
 * deterministic cigarette times inside the user's awake window.
 */
import { targetCountForDate } from './plan';
import { hashString, mulberry32 } from './rng';
import { addDays, zonedDateTimeToUtc } from './time';
import type { DayLog, PlanState, Settings, Slot } from './types';

const MINUTES_PER_DAY = 1440;
/** Keep the first/last slot this far from wake/sleep when the window allows. */
const EDGE_MARGIN_MINUTES = 15;

/**
 * Length of the awake window in minutes. When `sleep <= wake` the user goes to
 * bed after midnight, so the window wraps: sleep + 1440 − wake.
 */
export function awakeWindowMinutes(wakeMinutes: number, sleepMinutes: number): number {
  return sleepMinutes > wakeMinutes
    ? sleepMinutes - wakeMinutes
    : sleepMinutes + MINUTES_PER_DAY - wakeMinutes;
}

export type GenerateSlotsInput = {
  dateKey: string;
  seed: number;
  wakeMinutes: number;
  sleepMinutes: number;
  count: number;
  minGapMinutes: number;
};

/**
 * Generate `count` sorted minute-of-day values inside the awake window.
 *
 * Values are offsets from midnight of `dateKey`, so a wrapped window can yield
 * values above 1439 (those belong to the early hours of the next calendar day).
 *
 * Algorithm:
 *   W     = awake window length
 *   free  = W − (count − 1)·gap      (slack left after reserving the gaps)
 *   If free < 0 the requested gap is infeasible: gap := floor(W / (count − 1)).
 *   If count > W + 1 even a 1-minute gap is impossible: count := W + 1.
 *   An edge margin (≤ 15 min) is carved off both ends of `free` when possible.
 *   Draw `count` uniforms u_i in [0, free'], sort them, then
 *   slot_i = wake + margin + u_i + i·gap, rounded to whole minutes.
 *
 * Sorting the uniforms and adding i·gap guarantees consecutive slots differ by
 * at least `gap`. Rounding preserves this because `gap` is an integer and
 * Math.round is monotone. Deterministic for identical inputs.
 */
export function generateSlotMinutes(input: GenerateSlotsInput): number[] {
  const { dateKey, seed, wakeMinutes, sleepMinutes } = input;
  const W = awakeWindowMinutes(wakeMinutes, sleepMinutes);

  let count = Math.max(0, Math.floor(input.count));
  if (count === 0) return [];
  if (count > W + 1) count = W + 1;

  let gap = Math.max(0, Math.floor(input.minGapMinutes));
  let free = W - (count - 1) * gap;
  if (free < 0) {
    gap = count > 1 ? Math.floor(W / (count - 1)) : 0;
    free = W - (count - 1) * gap; // now >= 0
  }

  // Keep away from the edges of the window when there is slack to do so.
  const margin = Math.min(EDGE_MARGIN_MINUTES, Math.floor(free / 2));
  const usableFree = free - 2 * margin;

  const rng = mulberry32(hashString(`${seed}:${dateKey}`));
  const uniforms: number[] = [];
  for (let i = 0; i < count; i++) uniforms.push(rng() * usableFree);
  uniforms.sort((a, b) => a - b);

  return uniforms.map((u, i) => Math.round(wakeMinutes + margin + u + i * gap));
}

/** Build the full DayLog for `dateKey`: target from the plan, slots from the generator. */
export function buildDayLog(settings: Settings, plan: PlanState, dateKey: string): DayLog {
  const targetCount = targetCountForDate(settings, plan, dateKey);
  const minutes = generateSlotMinutes({
    dateKey,
    seed: plan.seed,
    wakeMinutes: settings.wakeMinutes,
    sleepMinutes: settings.sleepMinutes,
    count: targetCount,
    minGapMinutes: settings.minGapMinutes,
  });

  const slots: Slot[] = minutes.map((minuteOfDay, index) => {
    // Past-midnight slots belong to the next calendar day.
    const dayCarry = Math.floor(minuteOfDay / MINUTES_PER_DAY);
    const key = dayCarry === 0 ? dateKey : addDays(dateKey, dayCarry);
    const minute = minuteOfDay - dayCarry * MINUTES_PER_DAY;
    return {
      index,
      minuteOfDay,
      scheduledAtIso: zonedDateTimeToUtc(key, minute, settings.timeZone).toISOString(),
      status: 'pending',
    };
  });

  return { dateKey, targetCount, slots };
}

/**
 * Carry over slot statuses from a previously stored log onto a freshly built
 * one (matched by slot index). Used when logs are regenerated after a settings
 * change or on app foreground so the user's marks are not lost.
 */
export function mergeSlotStatuses(fresh: DayLog, existing: DayLog | null | undefined): DayLog {
  if (!existing || existing.dateKey !== fresh.dateKey) return fresh;
  const byIndex = new Map(existing.slots.map((s) => [s.index, s.status] as const));
  return {
    ...fresh,
    slots: fresh.slots.map((s) => ({ ...s, status: byIndex.get(s.index) ?? s.status })),
  };
}

/**
 * The earliest pending slot scheduled at or after `now`, across all logs.
 * Returns null when nothing is left.
 */
export function nextPendingSlot(logs: DayLog[], now: Date): Slot | null {
  const nowMs = now.getTime();
  let best: Slot | null = null;
  let bestMs = Infinity;
  for (const log of logs) {
    for (const slot of log.slots) {
      if (slot.status !== 'pending') continue;
      const t = Date.parse(slot.scheduledAtIso);
      if (t >= nowMs && t < bestMs) {
        best = slot;
        bestMs = t;
      }
    }
  }
  return best;
}
