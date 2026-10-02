/**
 * The taper plan: how many cigarettes are allowed on a given day.
 *
 * The plan starts at `startPerDay` and steps down by one every `daysPerStage`
 * days until it reaches 0 (quit). All functions are pure.
 */
import { addDays, daysBetween } from './time';
import type { PlanState, Settings } from './types';

/** Per-day counts for each stage, e.g. 5 → [5, 4, 3, 2, 1]. Stage after the last is quit (0). */
export function stageCounts(startPerDay: number): number[] {
  const n = Math.max(0, Math.floor(startPerDay));
  const counts: number[] = [];
  for (let c = n; c >= 1; c--) counts.push(c);
  return counts;
}

/** 0-based number of days since the plan started. Negative before the start date. */
export function dayIndex(plan: PlanState, dateKey: string): number {
  return daysBetween(plan.startDateKey, dateKey);
}

function safeDaysPerStage(settings: Settings): number {
  return Math.max(1, Math.floor(settings.daysPerStage));
}

/** Allowed cigarettes on `dateKey`: startPerDay − floor(dayIndex / daysPerStage), clamped at 0. */
export function targetCountForDate(settings: Settings, plan: PlanState, dateKey: string): number {
  const idx = Math.max(0, dayIndex(plan, dateKey));
  const stepsDown = Math.floor(idx / safeDaysPerStage(settings));
  return Math.max(0, Math.floor(settings.startPerDay) - stepsDown);
}

export type StageInfo = {
  /** 1-based stage number. The quit stage is `totalStages + 1`. */
  stageNumber: number;
  /** Number of smoking stages (= startPerDay). */
  totalStages: number;
  /** Cigarettes allowed per day in this stage (0 when quit). */
  perDay: number;
  /** 1-based day within the current stage (for the quit stage: days since quitting + 1). */
  dayInStage: number;
  daysPerStage: number;
  isQuit: boolean;
};

export function stageInfo(settings: Settings, plan: PlanState, dateKey: string): StageInfo {
  const dps = safeDaysPerStage(settings);
  const totalStages = Math.max(0, Math.floor(settings.startPerDay));
  const idx = Math.max(0, dayIndex(plan, dateKey));
  const perDay = targetCountForDate(settings, plan, dateKey);
  const isQuit = perDay === 0;
  const stageNumber = totalStages - perDay + 1;
  const dayInStage = isQuit ? idx - totalStages * dps + 1 : (idx % dps) + 1;
  return { stageNumber, totalStages, perDay, dayInStage, daysPerStage: dps, isQuit };
}

/** First dateKey on which the target is 0. */
export function quitDateKey(settings: Settings, plan: PlanState): string {
  const totalStages = Math.max(0, Math.floor(settings.startPerDay));
  return addDays(plan.startDateKey, totalStages * safeDaysPerStage(settings));
}
