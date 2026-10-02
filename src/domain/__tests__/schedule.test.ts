import {
  awakeWindowMinutes,
  buildDayLog,
  generateSlotMinutes,
  mergeSlotStatuses,
  nextPendingSlot,
} from '../schedule';
import { DEFAULT_SETTINGS, type DayLog, type PlanState, type Settings } from '../types';

const WAKE = 8 * 60; // 08:00
const SLEEP = 23 * 60; // 23:00

const base = { dateKey: '2026-10-02', seed: 1234, wakeMinutes: WAKE, sleepMinutes: SLEEP, count: 5, minGapMinutes: 30 };

function gaps(slots: number[]): number[] {
  return slots.slice(1).map((m, i) => m - slots[i]);
}

describe('awakeWindowMinutes', () => {
  it('measures a same-day window', () => {
    expect(awakeWindowMinutes(WAKE, SLEEP)).toBe(15 * 60);
  });

  it('wraps past midnight when sleep <= wake', () => {
    expect(awakeWindowMinutes(WAKE, 60)).toBe(17 * 60); // 08:00 → 01:00
    expect(awakeWindowMinutes(WAKE, WAKE)).toBe(1440);
  });
});

describe('generateSlotMinutes', () => {
  it('is deterministic for the same inputs', () => {
    expect(generateSlotMinutes(base)).toEqual(generateSlotMinutes({ ...base }));
  });

  it('changes with the seed and with the date', () => {
    const a = generateSlotMinutes(base);
    expect(generateSlotMinutes({ ...base, seed: 999 })).not.toEqual(a);
    expect(generateSlotMinutes({ ...base, dateKey: '2026-10-03' })).not.toEqual(a);
  });

  it('returns `count` sorted whole minutes', () => {
    const slots = generateSlotMinutes(base);
    expect(slots).toHaveLength(5);
    expect(slots).toEqual([...slots].sort((a, b) => a - b));
    slots.forEach((m) => expect(Number.isInteger(m)).toBe(true));
  });

  it('respects the minimum gap', () => {
    for (let seed = 0; seed < 50; seed++) {
      const slots = generateSlotMinutes({ ...base, seed, count: 8, minGapMinutes: 45 });
      gaps(slots).forEach((g) => expect(g).toBeGreaterThanOrEqual(45));
    }
  });

  it('keeps every slot inside the awake window with a 15-minute edge margin', () => {
    for (let seed = 0; seed < 50; seed++) {
      const slots = generateSlotMinutes({ ...base, seed });
      expect(slots[0]).toBeGreaterThanOrEqual(WAKE + 15);
      expect(slots[slots.length - 1]).toBeLessThanOrEqual(SLEEP - 15);
    }
  });

  it('handles a window that wraps past midnight', () => {
    const sleep = 2 * 60; // 02:00 next day
    for (let seed = 0; seed < 30; seed++) {
      const slots = generateSlotMinutes({ ...base, seed, sleepMinutes: sleep, count: 6 });
      expect(slots).toHaveLength(6);
      expect(slots[0]).toBeGreaterThanOrEqual(WAKE + 15);
      expect(slots[slots.length - 1]).toBeLessThanOrEqual(1440 + sleep - 15);
      gaps(slots).forEach((g) => expect(g).toBeGreaterThanOrEqual(30));
    }
    // Over many seeds at least one slot should land after midnight.
    const anyPastMidnight = Array.from({ length: 30 }, (_, seed) =>
      generateSlotMinutes({ ...base, seed, sleepMinutes: sleep, count: 6 }),
    ).some((s) => s.some((m) => m >= 1440));
    expect(anyPastMidnight).toBe(true);
  });

  it('reduces the gap when the requested one is infeasible', () => {
    // 2-hour window, 5 slots, 60-minute gap → impossible; gap shrinks to 30.
    const slots = generateSlotMinutes({ ...base, wakeMinutes: 600, sleepMinutes: 720, count: 5, minGapMinutes: 60 });
    expect(slots).toHaveLength(5);
    expect(slots[0]).toBeGreaterThanOrEqual(600);
    expect(slots[slots.length - 1]).toBeLessThanOrEqual(720);
    gaps(slots).forEach((g) => expect(g).toBeGreaterThanOrEqual(30));
  });

  it('clamps the count when even 1-minute gaps do not fit', () => {
    const slots = generateSlotMinutes({ ...base, wakeMinutes: 600, sleepMinutes: 603, count: 10, minGapMinutes: 30 });
    expect(slots).toEqual([600, 601, 602, 603]);
  });

  it('returns an empty array for a zero count', () => {
    expect(generateSlotMinutes({ ...base, count: 0 })).toEqual([]);
  });
});

describe('buildDayLog', () => {
  const settings: Settings = { ...DEFAULT_SETTINGS, timeZone: 'Asia/Jerusalem', wakeMinutes: WAKE, sleepMinutes: SLEEP };
  const plan: PlanState = { startDateKey: '2026-10-02', seed: 7 };

  it('uses the plan target and produces pending slots with UTC instants', () => {
    const log = buildDayLog(settings, plan, '2026-10-02');
    expect(log.dateKey).toBe('2026-10-02');
    expect(log.targetCount).toBe(5);
    expect(log.slots).toHaveLength(5);
    log.slots.forEach((slot, i) => {
      expect(slot.index).toBe(i);
      expect(slot.status).toBe('pending');
      // 08:00 Jerusalem on 2026-10-02 is 05:00Z; sleep 23:00 is 20:00Z.
      const t = Date.parse(slot.scheduledAtIso);
      expect(t).toBeGreaterThanOrEqual(Date.parse('2026-10-02T05:00:00Z'));
      expect(t).toBeLessThanOrEqual(Date.parse('2026-10-02T20:00:00Z'));
    });
  });

  it('places past-midnight slots on the next calendar day', () => {
    const late: Settings = { ...settings, wakeMinutes: 20 * 60, sleepMinutes: 3 * 60, minGapMinutes: 60 };
    const log = buildDayLog(late, plan, '2026-10-02');
    const wrapped = log.slots.filter((s) => s.minuteOfDay >= 1440);
    expect(wrapped.length).toBeGreaterThan(0);
    wrapped.forEach((s) => {
      // 00:00 Jerusalem on Oct 3 == 21:00Z on Oct 2.
      expect(Date.parse(s.scheduledAtIso)).toBeGreaterThanOrEqual(Date.parse('2026-10-02T21:00:00Z'));
      expect(Date.parse(s.scheduledAtIso)).toBeLessThanOrEqual(Date.parse('2026-10-03T00:00:00Z'));
    });
  });

  it('builds an empty log on the quit day', () => {
    const log = buildDayLog(settings, plan, '2026-11-06');
    expect(log.targetCount).toBe(0);
    expect(log.slots).toEqual([]);
  });
});

describe('mergeSlotStatuses', () => {
  it('carries statuses over by index', () => {
    const fresh: DayLog = {
      dateKey: 'd',
      targetCount: 2,
      slots: [
        { index: 0, minuteOfDay: 500, scheduledAtIso: 'a', status: 'pending' },
        { index: 1, minuteOfDay: 600, scheduledAtIso: 'b', status: 'pending' },
      ],
    };
    const existing: DayLog = { ...fresh, slots: [{ ...fresh.slots[0], status: 'smoked' }] };
    const merged = mergeSlotStatuses(fresh, existing);
    expect(merged.slots.map((s) => s.status)).toEqual(['smoked', 'pending']);
    expect(mergeSlotStatuses(fresh, null)).toBe(fresh);
  });
});

describe('nextPendingSlot', () => {
  const log: DayLog = {
    dateKey: '2026-10-02',
    targetCount: 3,
    slots: [
      { index: 0, minuteOfDay: 540, scheduledAtIso: '2026-10-02T06:00:00.000Z', status: 'smoked' },
      { index: 1, minuteOfDay: 720, scheduledAtIso: '2026-10-02T09:00:00.000Z', status: 'pending' },
      { index: 2, minuteOfDay: 900, scheduledAtIso: '2026-10-02T12:00:00.000Z', status: 'pending' },
    ],
  };

  it('returns the earliest pending slot at or after now', () => {
    expect(nextPendingSlot([log], new Date('2026-10-02T05:00:00Z'))?.index).toBe(1);
    expect(nextPendingSlot([log], new Date('2026-10-02T09:00:00Z'))?.index).toBe(1);
    expect(nextPendingSlot([log], new Date('2026-10-02T10:00:00Z'))?.index).toBe(2);
  });

  it('returns null when nothing is left', () => {
    expect(nextPendingSlot([log], new Date('2026-10-02T13:00:00Z'))).toBeNull();
    expect(nextPendingSlot([], new Date())).toBeNull();
  });
});
