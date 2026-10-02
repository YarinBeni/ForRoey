/**
 * Typed AsyncStorage wrapper.
 *
 * All keys are prefixed with `pacer:`. Every loader returns a sensible default
 * (or null) when nothing is stored or the stored JSON is corrupt, so callers
 * never have to deal with parse errors.
 */
import AsyncStorage from '@react-native-async-storage/async-storage';

import { DEFAULT_SETTINGS, type DayLog, type PlanState, type Settings } from '../domain/types';

const PREFIX = 'pacer:';

export const STORAGE_KEYS = {
  settings: `${PREFIX}settings`,
  plan: `${PREFIX}plan`,
  trialStart: `${PREFIX}trialStart`,
  dayLog: (dateKey: string) => `${PREFIX}dayLog:${dateKey}`,
} as const;

async function readJson<T>(key: string): Promise<T | null> {
  try {
    const raw = await AsyncStorage.getItem(key);
    if (raw == null) return null;
    return JSON.parse(raw) as T;
  } catch (err) {
    console.warn(`[storage] failed to read ${key}`, err);
    return null;
  }
}

async function writeJson(key: string, value: unknown): Promise<void> {
  await AsyncStorage.setItem(key, JSON.stringify(value));
}

// ---------------------------------------------------------------------------
// Settings
// ---------------------------------------------------------------------------

/** Stored settings merged over DEFAULT_SETTINGS so new fields always have values. */
export async function loadSettings(): Promise<Settings> {
  const stored = await readJson<Partial<Settings>>(STORAGE_KEYS.settings);
  return { ...DEFAULT_SETTINGS, ...(stored ?? {}) };
}

export async function saveSettings(settings: Settings): Promise<void> {
  await writeJson(STORAGE_KEYS.settings, settings);
}

// ---------------------------------------------------------------------------
// Plan
// ---------------------------------------------------------------------------

/** null means the user has not completed onboarding yet. */
export async function loadPlan(): Promise<PlanState | null> {
  const plan = await readJson<PlanState>(STORAGE_KEYS.plan);
  if (!plan || typeof plan.startDateKey !== 'string' || typeof plan.seed !== 'number') {
    return null;
  }
  return plan;
}

export async function savePlan(plan: PlanState): Promise<void> {
  await writeJson(STORAGE_KEYS.plan, plan);
}

export async function clearPlan(): Promise<void> {
  await AsyncStorage.removeItem(STORAGE_KEYS.plan);
}

// ---------------------------------------------------------------------------
// Day logs
// ---------------------------------------------------------------------------

export async function loadDayLog(dateKey: string): Promise<DayLog | null> {
  const log = await readJson<DayLog>(STORAGE_KEYS.dayLog(dateKey));
  if (!log || log.dateKey !== dateKey || !Array.isArray(log.slots)) return null;
  return log;
}

export async function saveDayLog(log: DayLog): Promise<void> {
  await writeJson(STORAGE_KEYS.dayLog(log.dateKey), log);
}

/** Load several day logs at once; missing days come back as null in the same order. */
export async function loadDayLogs(dateKeys: string[]): Promise<(DayLog | null)[]> {
  return Promise.all(dateKeys.map((k) => loadDayLog(k)));
}

// ---------------------------------------------------------------------------
// Trial
// ---------------------------------------------------------------------------

/** ISO timestamp of when the free trial began, or null if never recorded. */
export async function loadTrialStart(): Promise<string | null> {
  const value = await readJson<string>(STORAGE_KEYS.trialStart);
  return typeof value === 'string' && !Number.isNaN(Date.parse(value)) ? value : null;
}

export async function saveTrialStart(iso: string): Promise<void> {
  await writeJson(STORAGE_KEYS.trialStart, iso);
}

/** Remove every `pacer:` key. Intended for a "reset app" action and tests. */
export async function clearAll(): Promise<void> {
  const keys = await AsyncStorage.getAllKeys();
  const ours = keys.filter((k) => k.startsWith(PREFIX));
  if (ours.length > 0) await AsyncStorage.multiRemove(ours);
}
