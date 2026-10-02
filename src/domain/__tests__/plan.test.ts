import { dayIndex, quitDateKey, stageCounts, stageInfo, targetCountForDate } from '../plan';
import { addDays } from '../time';
import { DEFAULT_SETTINGS, type PlanState, type Settings } from '../types';

const settings: Settings = { ...DEFAULT_SETTINGS, startPerDay: 5, daysPerStage: 7 };
const plan: PlanState = { startDateKey: '2026-10-02', seed: 42 };

describe('stageCounts', () => {
  it('lists per-day counts down to 1', () => {
    expect(stageCounts(5)).toEqual([5, 4, 3, 2, 1]);
    expect(stageCounts(1)).toEqual([1]);
    expect(stageCounts(0)).toEqual([]);
  });
});

describe('targetCountForDate', () => {
  it('tapers 5→4→3→2→1→0 every daysPerStage days', () => {
    const expectedByDay: number[] = [];
    for (let d = 0; d < 7 * 6; d++) expectedByDay.push(Math.max(0, 5 - Math.floor(d / 7)));

    expectedByDay.forEach((expected, d) => {
      expect(targetCountForDate(settings, plan, addDays(plan.startDateKey, d))).toBe(expected);
    });

    expect(targetCountForDate(settings, plan, addDays(plan.startDateKey, 0))).toBe(5);
    expect(targetCountForDate(settings, plan, addDays(plan.startDateKey, 6))).toBe(5);
    expect(targetCountForDate(settings, plan, addDays(plan.startDateKey, 7))).toBe(4);
    expect(targetCountForDate(settings, plan, addDays(plan.startDateKey, 28))).toBe(1);
    expect(targetCountForDate(settings, plan, addDays(plan.startDateKey, 35))).toBe(0);
    expect(targetCountForDate(settings, plan, addDays(plan.startDateKey, 400))).toBe(0);
  });

  it('treats dates before the start as day 0', () => {
    expect(dayIndex(plan, '2026-10-01')).toBe(-1);
    expect(targetCountForDate(settings, plan, '2026-10-01')).toBe(5);
  });

  it('respects a different daysPerStage', () => {
    const s3 = { ...settings, daysPerStage: 3 };
    expect(targetCountForDate(s3, plan, addDays(plan.startDateKey, 2))).toBe(5);
    expect(targetCountForDate(s3, plan, addDays(plan.startDateKey, 3))).toBe(4);
    expect(targetCountForDate(s3, plan, addDays(plan.startDateKey, 15))).toBe(0);
  });
});

describe('quitDateKey', () => {
  it('is the first day with a zero target', () => {
    const quit = quitDateKey(settings, plan);
    expect(quit).toBe('2026-11-06'); // 35 days after 2026-10-02
    expect(targetCountForDate(settings, plan, quit)).toBe(0);
    expect(targetCountForDate(settings, plan, addDays(quit, -1))).toBe(1);
  });
});

describe('stageInfo', () => {
  it('describes the first day', () => {
    expect(stageInfo(settings, plan, plan.startDateKey)).toEqual({
      stageNumber: 1,
      totalStages: 5,
      perDay: 5,
      dayInStage: 1,
      daysPerStage: 7,
      isQuit: false,
    });
  });

  it('describes a mid-plan day', () => {
    const info = stageInfo(settings, plan, addDays(plan.startDateKey, 9)); // stage 2, day 3
    expect(info.stageNumber).toBe(2);
    expect(info.perDay).toBe(4);
    expect(info.dayInStage).toBe(3);
    expect(info.isQuit).toBe(false);
  });

  it('describes the quit stage', () => {
    const info = stageInfo(settings, plan, addDays(quitDateKey(settings, plan), 1));
    expect(info.isQuit).toBe(true);
    expect(info.perDay).toBe(0);
    expect(info.stageNumber).toBe(6);
    expect(info.dayInStage).toBe(2);
  });
});
