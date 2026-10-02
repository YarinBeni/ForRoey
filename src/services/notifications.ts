/**
 * Local notification scheduling for cigarette slots (expo-notifications).
 *
 * Strategy: on every relevant change we cancel everything and re-schedule the
 * upcoming pending slots. That keeps the OS queue an exact mirror of the
 * current plan without having to track notification identifiers.
 */
import * as Device from 'expo-device';
import * as Notifications from 'expo-notifications';
import { Platform } from 'react-native';

import type { DayLog, Slot } from '../domain/types';

/** Android notification channel used for all slot reminders. */
export const SLOTS_CHANNEL_ID = 'slots';

/** iOS allows 64 pending local notifications per app; leave headroom. */
export const MAX_SCHEDULED = 60;

/**
 * Make notifications visible while the app is in the foreground and create
 * the Android channel. Call once at app start (e.g. in the root layout).
 */
export function configureNotificationHandler(): void {
  Notifications.setNotificationHandler({
    handleNotification: async () => ({
      shouldShowBanner: true,
      shouldShowList: true,
      shouldPlaySound: true,
      shouldSetBadge: false,
    }),
  });

  if (Platform.OS === 'android') {
    Notifications.setNotificationChannelAsync(SLOTS_CHANNEL_ID, {
      name: 'Cigarette slots',
      description: 'Reminds you when your next planned cigarette is available.',
      importance: Notifications.AndroidImportance.HIGH,
      sound: 'default',
      lightColor: '#157A6E',
    }).catch((err) => console.warn('[notifications] channel setup failed', err));
  }
}

/** Ask for permission if needed. Resolves true when notifications may be shown. */
export async function requestNotificationPermission(): Promise<boolean> {
  try {
    const current = await Notifications.getPermissionsAsync();
    if (current.granted) return true;
    // On iOS `canAskAgain` is false once the user has declined in the system prompt.
    if (current.canAskAgain === false) return false;
    const requested = await Notifications.requestPermissionsAsync({
      ios: { allowAlert: true, allowSound: true, allowBadge: false },
    });
    return requested.granted;
  } catch (err) {
    console.warn('[notifications] permission request failed', err);
    return false;
  }
}

type PendingSlotRef = { log: DayLog; slot: Slot; atMs: number };

/**
 * Replace all scheduled notifications with one per pending, future slot in
 * `dayLogs` (earliest first, capped at MAX_SCHEDULED).
 *
 * @param dayLogs    Logs to schedule from (typically today .. today+6).
 * @param gapMinutes The user's minimum gap, shown in the notification body.
 * @returns number of notifications scheduled.
 */
export async function syncScheduledNotifications(
  dayLogs: DayLog[],
  gapMinutes: number,
): Promise<number> {
  if (!Device.isDevice) {
    console.log('[notifications] running on a simulator/emulator; skipping scheduling');
    return 0;
  }

  try {
    await Notifications.cancelAllScheduledNotificationsAsync();
  } catch (err) {
    console.warn('[notifications] cancelAll failed', err);
  }

  const now = Date.now();
  const upcoming: PendingSlotRef[] = [];
  for (const log of dayLogs) {
    for (const slot of log.slots) {
      if (slot.status !== 'pending') continue;
      const atMs = Date.parse(slot.scheduledAtIso);
      if (Number.isFinite(atMs) && atMs > now) upcoming.push({ log, slot, atMs });
    }
  }
  upcoming.sort((a, b) => a.atMs - b.atMs);

  const toSchedule = upcoming.slice(0, MAX_SCHEDULED);
  let scheduled = 0;
  for (const { log, slot, atMs } of toSchedule) {
    try {
      await Notifications.scheduleNotificationAsync({
        content: {
          title: 'You can have a cigarette now',
          body: `Slot ${slot.index + 1} of ${log.targetCount} today · next one in at least ${gapMinutes} min`,
          sound: 'default',
          data: { dateKey: log.dateKey, index: slot.index },
        },
        trigger: {
          type: Notifications.SchedulableTriggerInputTypes.DATE,
          date: new Date(atMs),
          channelId: SLOTS_CHANNEL_ID,
        },
      });
      scheduled++;
    } catch (err) {
      console.warn(`[notifications] failed to schedule ${log.dateKey}#${slot.index}`, err);
    }
  }
  return scheduled;
}

/** Cancel every scheduled slot notification (used when the user turns reminders off). */
export async function cancelAllSlotNotifications(): Promise<void> {
  try {
    await Notifications.cancelAllScheduledNotificationsAsync();
  } catch (err) {
    console.warn('[notifications] cancelAll failed', err);
  }
}
