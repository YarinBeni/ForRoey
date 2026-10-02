/**
 * Subscription / paywall abstraction.
 *
 * Two implementations share one interface:
 *  - LocalTrialService: a device-local 30-day trial, used in Expo Go and
 *    whenever RevenueCat keys are not configured. Purchases are not possible.
 *  - RevenueCatService: wraps react-native-purchases; grants `active` when the
 *    "pro" entitlement is present and otherwise falls back to the local trial.
 */
import Constants from 'expo-constants';
import { Platform } from 'react-native';
import Purchases, { type CustomerInfo, type PurchasesPackage } from 'react-native-purchases';

import { loadTrialStart, saveTrialStart } from './storage';

export type SubscriptionStatus =
  | { kind: 'trial'; trialEndsAt: string; daysLeft: number }
  | { kind: 'active' }
  | { kind: 'expired' };

export interface SubscriptionService {
  /** Prepare the service (configure SDKs, start the trial clock). Safe to call once at startup. */
  init(): Promise<void>;
  getStatus(): Promise<SubscriptionStatus>;
  /** Buy the monthly subscription. Resolves with the new status; rejects on failure/cancel. */
  purchaseMonthly(): Promise<SubscriptionStatus>;
  /** Restore previous purchases (required by App Store review). */
  restore(): Promise<SubscriptionStatus>;
  /** Human-readable price, e.g. "₪10 / month". */
  readonly priceLabel: string;
}

/** Entitlement identifier configured in the RevenueCat dashboard. */
export const PRO_ENTITLEMENT_ID = 'pro';

const MS_PER_DAY = 86_400_000;

type ExtraConfig = {
  revenueCatIosApiKey?: string;
  revenueCatAndroidApiKey?: string;
  trialDays?: number;
  priceLabel?: string;
};

function readExtra(): ExtraConfig {
  return (Constants.expoConfig?.extra ?? {}) as ExtraConfig;
}

/** Length of the free trial in days, from app.json `extra.trialDays` (default 30). */
export function trialDaysFromConfig(): number {
  const d = readExtra().trialDays;
  return typeof d === 'number' && d > 0 ? d : 30;
}

function priceLabelFromConfig(): string {
  return readExtra().priceLabel ?? '₪10 / month';
}

/** Compute a trial/expired status from the trial start instant. */
export function trialStatusFrom(trialStartIso: string, trialDays: number, now = new Date()): SubscriptionStatus {
  const endMs = Date.parse(trialStartIso) + trialDays * MS_PER_DAY;
  const daysLeft = Math.ceil((endMs - now.getTime()) / MS_PER_DAY);
  if (daysLeft <= 0) return { kind: 'expired' };
  return { kind: 'trial', trialEndsAt: new Date(endMs).toISOString(), daysLeft };
}

// ---------------------------------------------------------------------------
// Local trial
// ---------------------------------------------------------------------------

export class LocalTrialService implements SubscriptionService {
  readonly priceLabel = priceLabelFromConfig();

  private readonly trialDays = trialDaysFromConfig();

  async init(): Promise<void> {
    // Start the trial clock the first time the app runs.
    const existing = await loadTrialStart();
    if (!existing) await saveTrialStart(new Date().toISOString());
  }

  async getStatus(): Promise<SubscriptionStatus> {
    let start = await loadTrialStart();
    if (!start) {
      start = new Date().toISOString();
      await saveTrialStart(start);
    }
    return trialStatusFrom(start, this.trialDays);
  }

  async purchaseMonthly(): Promise<SubscriptionStatus> {
    throw new Error(
      'Store not configured: add RevenueCat API keys to app.json → expo.extra and run a development build.',
    );
  }

  async restore(): Promise<SubscriptionStatus> {
    throw new Error('Store not configured: purchases cannot be restored in this build.');
  }
}

// ---------------------------------------------------------------------------
// RevenueCat
// ---------------------------------------------------------------------------

export class RevenueCatService implements SubscriptionService {
  readonly priceLabel = priceLabelFromConfig();

  private readonly trial = new LocalTrialService();

  private configured = false;

  constructor(private readonly apiKey: string) {}

  async init(): Promise<void> {
    await this.trial.init();
    try {
      Purchases.configure({ apiKey: this.apiKey });
      this.configured = true;
    } catch (err) {
      console.warn('[subscription] RevenueCat configure failed; falling back to trial', err);
      this.configured = false;
    }
  }

  private hasPro(info: CustomerInfo): boolean {
    return Boolean(info.entitlements.active[PRO_ENTITLEMENT_ID]);
  }

  /** `active` when entitled, otherwise whatever the local trial says. */
  private async statusFrom(info: CustomerInfo | null): Promise<SubscriptionStatus> {
    if (info && this.hasPro(info)) return { kind: 'active' };
    return this.trial.getStatus();
  }

  async getStatus(): Promise<SubscriptionStatus> {
    if (!this.configured) return this.trial.getStatus();
    try {
      const info = await Purchases.getCustomerInfo();
      return this.statusFrom(info);
    } catch (err) {
      console.warn('[subscription] getCustomerInfo failed', err);
      return this.trial.getStatus();
    }
  }

  private async monthlyPackage(): Promise<PurchasesPackage> {
    const offerings = await Purchases.getOfferings();
    const current = offerings.current;
    const pkg = current?.monthly ?? current?.availablePackages[0] ?? null;
    if (!pkg) throw new Error('No monthly package is available in the current offering.');
    return pkg;
  }

  async purchaseMonthly(): Promise<SubscriptionStatus> {
    if (!this.configured) throw new Error('Store not configured.');
    try {
      const pkg = await this.monthlyPackage();
      const { customerInfo } = await Purchases.purchasePackage(pkg);
      return this.statusFrom(customerInfo);
    } catch (err) {
      // RevenueCat marks user cancellations; surface a friendlier message.
      if (typeof err === 'object' && err !== null && (err as { userCancelled?: boolean }).userCancelled) {
        throw new Error('Purchase cancelled.');
      }
      throw err instanceof Error ? err : new Error('Purchase failed.');
    }
  }

  async restore(): Promise<SubscriptionStatus> {
    if (!this.configured) throw new Error('Store not configured.');
    try {
      const info = await Purchases.restorePurchases();
      return this.statusFrom(info);
    } catch (err) {
      throw err instanceof Error ? err : new Error('Restore failed.');
    }
  }
}

// ---------------------------------------------------------------------------
// Factory
// ---------------------------------------------------------------------------

function platformApiKey(): string {
  const extra = readExtra();
  if (Platform.OS === 'ios') return extra.revenueCatIosApiKey ?? '';
  if (Platform.OS === 'android') return extra.revenueCatAndroidApiKey ?? '';
  return '';
}

/**
 * RevenueCat when a platform API key is set and we are not running inside
 * Expo Go (which lacks the native module); otherwise the local trial.
 */
export function createSubscriptionService(): SubscriptionService {
  const key = platformApiKey();
  const inExpoGo = Constants.appOwnership === 'expo';
  if (key.length > 0 && !inExpoGo) return new RevenueCatService(key);
  return new LocalTrialService();
}
