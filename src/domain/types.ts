/**
 * Core domain types for Pacer.
 *
 * Everything here is plain data (JSON-serialisable) so it can be persisted to
 * AsyncStorage and reasoned about in pure functions without React or React
 * Native imports.
 */

export type Settings = {
  /** IANA time zone the user's day is computed in, e.g. "Asia/Jerusalem". */
  timeZone: string;
  /** Wake-up time as minutes after midnight, 0..1439. */
  wakeMinutes: number;
  /**
   * Bedtime as minutes after midnight, 0..1439. May be <= wakeMinutes, which
   * means the user goes to sleep after midnight (the awake window wraps).
   */
  sleepMinutes: number;
  /** Minimum number of minutes between two cigarette slots. Default 30. */
  minGapMinutes: number;
  /** Cigarettes per day on the first stage of the plan. Default 5. */
  startPerDay: number;
  /** How many days to stay at each per-day count before stepping down. Default 7. */
  daysPerStage: number;
  /** Whether slot reminders should be scheduled. */
  notificationsEnabled: boolean;
};

export type SlotStatus = 'pending' | 'smoked' | 'skipped';

export type Slot = {
  /** 0-based position of the slot within its day. */
  index: number;
  /**
   * Minutes after midnight of the day's `dateKey` in the user's time zone.
   * Can exceed 1439 when the slot falls after midnight (wrapped awake window).
   */
  minuteOfDay: number;
  /** The exact instant of the slot as an ISO-8601 UTC string. */
  scheduledAtIso: string;
  status: SlotStatus;
};

export type DayLog = {
  /** "YYYY-MM-DD" in the user's time zone. */
  dateKey: string;
  /** How many cigarettes the plan allows on this day. */
  targetCount: number;
  slots: Slot[];
};

export type PlanState = {
  /** The dateKey ("YYYY-MM-DD" in the user's tz) of day 0 of the plan. */
  startDateKey: string;
  /** Random seed fixed at onboarding so slot times are stable across launches. */
  seed: number;
};

/** A "YYYY-MM-DD" string in the user's time zone. */
export type DateKey = string;

function detectTimeZone(): string {
  try {
    const tz = Intl.DateTimeFormat().resolvedOptions().timeZone;
    if (tz && tz.length > 0) return tz;
  } catch {
    // Intl may be unavailable or throw on some exotic runtimes; fall through.
  }
  return 'Asia/Jerusalem';
}

export const DEFAULT_SETTINGS: Settings = {
  timeZone: detectTimeZone(),
  wakeMinutes: 8 * 60, // 08:00
  sleepMinutes: 23 * 60, // 23:00
  minGapMinutes: 30,
  startPerDay: 5,
  daysPerStage: 7,
  notificationsEnabled: true,
};
