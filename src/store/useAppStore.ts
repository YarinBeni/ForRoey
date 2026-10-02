/**
 * Global app state (zustand).
 *
 * The store owns persistence and side effects (storage, notifications,
 * subscription) so that screens stay thin. It deliberately imports nothing
 * from React Native's UI layer.
 */
import { create } from 'zustand';

import { buildDayLog, mergeSlotStatuses } from '../domain/schedule';
import { addDays, dateKeyInTz } from '../domain/time';
import { DEFAULT_SETTINGS, type DayLog, type PlanState, type Settings, type SlotStatus } from '../domain/types';
import { cancelAllSlotNotifications, syncScheduledNotifications } from '../services/notifications';
import {
  loadDayLog,
  loadPlan,
  loadSettings,
  saveDayLog,
  savePlan,
  saveSettings,
} from '../services/storage';
import {
  createSubscriptionService,
  type SubscriptionService,
  type SubscriptionStatus,
} from '../services/subscription';

/** How many days ahead (including today) we materialise logs and notifications for. */
export const LOOKAHEAD_DAYS = 7;

export type AppState = {
  /** True once persisted state has been loaded. Screens should wait for this. */
  hydrated: boolean;
  /** True once a plan exists (onboarding finished). */
  onboarded: boolean;
  settings: Settings;
  plan: PlanState | null;
  todayLog: DayLog | null;
  tomorrowLog: DayLog | null;
  subscription: SubscriptionStatus | null;

  hydrate(): Promise<void>;
  completeOnboarding(settings: Settings): Promise<void>;
  /** Change settings in memory only (onboarding), without persisting or rescheduling. */
  setDraftSettings(partial: Partial<Settings>): void;
  updateSettings(partial: Partial<Settings>): Promise<void>;
  /** Start the step-down plan again from today, discarding today's marks. */
  restartPlan(): Promise<void>;
  markSlot(dateKey: string, index: number, status: SlotStatus): Promise<void>;
  refreshToday(): Promise<void>;
  refreshSubscription(): Promise<void>;
};

let subscriptionService: SubscriptionService | null = null;

/** Lazily create the subscription service so importing the store has no side effects. */
export function getSubscriptionService(): SubscriptionService {
  if (!subscriptionService) subscriptionService = createSubscriptionService();
  return subscriptionService;
}

function todayKey(settings: Settings): string {
  return dateKeyInTz(new Date(), settings.timeZone);
}

/**
 * Rebuild and persist the logs for today .. today+LOOKAHEAD_DAYS−1, carrying
 * over existing slot statuses, then mirror them into the notification queue.
 * Returns the window of logs (index 0 = today).
 */
async function rebuildWindow(settings: Settings, plan: PlanState, carryStatuses = true): Promise<DayLog[]> {
  const start = todayKey(settings);
  const logs: DayLog[] = [];
  for (let i = 0; i < LOOKAHEAD_DAYS; i++) {
    const key = addDays(start, i);
    const fresh = buildDayLog(settings, plan, key);
    const merged = carryStatuses ? mergeSlotStatuses(fresh, await loadDayLog(key)) : fresh;
    await saveDayLog(merged);
    logs.push(merged);
  }
  await syncNotifications(settings, logs);
  return logs;
}

async function syncNotifications(settings: Settings, logs: DayLog[]): Promise<void> {
  try {
    if (settings.notificationsEnabled) {
      await syncScheduledNotifications(logs, settings.minGapMinutes);
    } else {
      await cancelAllSlotNotifications();
    }
  } catch (err) {
    console.warn('[store] notification sync failed', err);
  }
}

/** Load the current window of logs from storage (without regenerating). */
async function loadWindow(settings: Settings): Promise<DayLog[]> {
  const start = todayKey(settings);
  const logs: DayLog[] = [];
  for (let i = 0; i < LOOKAHEAD_DAYS; i++) {
    const log = await loadDayLog(addDays(start, i));
    if (log) logs.push(log);
  }
  return logs;
}

function randomSeed(): number {
  // Math.random is fine here: the seed is generated once and then persisted.
  return Math.floor(Math.random() * 0xffffffff) >>> 0;
}

export const useAppStore = create<AppState>()((set, get) => ({
  hydrated: false,
  onboarded: false,
  settings: DEFAULT_SETTINGS,
  plan: null,
  todayLog: null,
  tomorrowLog: null,
  subscription: null,

  async hydrate() {
    const [settings, plan] = await Promise.all([loadSettings(), loadPlan()]);

    let todayLog: DayLog | null = null;
    let tomorrowLog: DayLog | null = null;
    if (plan) {
      const logs = await rebuildWindow(settings, plan);
      todayLog = logs[0] ?? null;
      tomorrowLog = logs[1] ?? null;
    }

    let subscription: SubscriptionStatus | null = null;
    try {
      const service = getSubscriptionService();
      await service.init();
      subscription = await service.getStatus();
    } catch (err) {
      console.warn('[store] subscription init failed', err);
    }

    set({ hydrated: true, onboarded: plan !== null, settings, plan, todayLog, tomorrowLog, subscription });
  },

  async completeOnboarding(settings) {
    const plan: PlanState = { startDateKey: todayKey(settings), seed: randomSeed() };
    await Promise.all([saveSettings(settings), savePlan(plan)]);
    const logs = await rebuildWindow(settings, plan);
    set({ onboarded: true, settings, plan, todayLog: logs[0] ?? null, tomorrowLog: logs[1] ?? null });
  },

  setDraftSettings(partial) {
    set({ settings: { ...get().settings, ...partial } });
  },

  async updateSettings(partial) {
    const settings: Settings = { ...get().settings, ...partial };
    await saveSettings(settings);
    const { plan } = get();
    if (!plan) {
      set({ settings });
      return;
    }
    const logs = await rebuildWindow(settings, plan);
    set({ settings, todayLog: logs[0] ?? null, tomorrowLog: logs[1] ?? null });
  },

  async markSlot(dateKey, index, status) {
    const { settings, todayLog, tomorrowLog } = get();
    const inMemory = [todayLog, tomorrowLog].find((l) => l?.dateKey === dateKey) ?? null;
    const log = inMemory ?? (await loadDayLog(dateKey));
    if (!log) return;

    const updated: DayLog = {
      ...log,
      slots: log.slots.map((s) => (s.index === index ? { ...s, status } : s)),
    };
    await saveDayLog(updated);

    set({
      todayLog: todayLog?.dateKey === dateKey ? updated : todayLog,
      tomorrowLog: tomorrowLog?.dateKey === dateKey ? updated : tomorrowLog,
    });

    // A smoked/skipped slot no longer needs a reminder.
    await syncNotifications(settings, await loadWindow(settings));
  },

  async restartPlan() {
    const { settings } = get();
    const plan: PlanState = { startDateKey: todayKey(settings), seed: randomSeed() };
    await savePlan(plan);
    const logs = await rebuildWindow(settings, plan, false);
    set({ plan, todayLog: logs[0] ?? null, tomorrowLog: logs[1] ?? null });
  },

  async refreshToday() {
    const { settings, plan } = get();
    if (!plan) return;
    const logs = await rebuildWindow(settings, plan);
    set({ todayLog: logs[0] ?? null, tomorrowLog: logs[1] ?? null });
  },

  async refreshSubscription() {
    try {
      const subscription = await getSubscriptionService().getStatus();
      set({ subscription });
    } catch (err) {
      console.warn('[store] subscription refresh failed', err);
    }
  },
}));
