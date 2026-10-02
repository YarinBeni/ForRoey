import { requireOptionalNativeModule } from 'expo';
import { Platform } from 'react-native';

/**
 * Burning-cigarette status bar icon (Android only).
 *
 * While a smoking window is open, a silent ongoing notification shows a
 * cigarette in the status bar that burns down to the filter as the window
 * passes. iOS has no app icons in its status bar, so these calls are no-ops
 * there, as they are in Expo Go where the native module is not compiled in.
 */
export type StatusWindow = {
  /** Stable id, e.g. `${dateKey}:${index}`. */
  id: string;
  /** Epoch milliseconds when the window opens. */
  openAtMs: number;
  /** Length of the window in milliseconds. */
  windowMs: number;
};

type NativeModule = {
  scheduleWindows(windows: StatusWindow[]): void;
  cancelAll(): void;
  clear(): void;
  preview(windowMs: number): void;
  canScheduleExactAlarms(): boolean;
};

const native: NativeModule | null =
  Platform.OS === 'android' ? requireOptionalNativeModule<NativeModule>('CigaretteStatus') : null;

/** True when the device can show the icon (Android with the native module built in). */
export const statusIconSupported = native !== null;

export function scheduleStatusWindows(windows: StatusWindow[]): void {
  native?.scheduleWindows(windows);
}

export function cancelAllStatusWindows(): void {
  native?.cancelAll();
}

/** Hide the icon for the window that is open right now. */
export function clearStatusIcon(): void {
  native?.clear();
}

export function previewStatusIcon(windowMs: number): void {
  native?.preview(windowMs);
}

export function canScheduleExactAlarms(): boolean {
  return native?.canScheduleExactAlarms() ?? false;
}
