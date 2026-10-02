# Pacer

A small Android-first app (iOS-ready) that helps you smoke less, then stop.

You tell Pacer when you are awake. Every day it picks random moments inside that window, at least 30 minutes apart, and sends you a notification: *you can have a cigarette now*. While the 30-minute window is open, a cigarette icon in the phone's status bar burns down until it is gone. You start at 5 a day. Every week the count drops by one, until it reaches zero.

Built with Expo (React Native + TypeScript), Expo Router, local notifications, a small Kotlin module for the status-bar icon, and a RevenueCat-backed monthly subscription with a free first month.

## How it works

| Piece | Where | What it does |
| --- | --- | --- |
| Schedule engine | `src/domain/schedule.ts` | Draws N random slot times inside the awake window with a minimum gap. Seeded by the plan and the date, so re-opening the app never moves today's times. |
| Step-down plan | `src/domain/plan.ts` | 5 → 4 → 3 → 2 → 1 → 0 a day, one step every `daysPerStage` days (default 7). Computes the quit date. |
| Time zones | `src/domain/time.ts` | Wall-clock ↔ instant conversion with `Intl` only, DST-safe. |
| Notifications | `src/services/notifications.ts` | Schedules the next 7 days of slots as local notifications. Re-synced every time the app comes to the foreground. |
| Status-bar cigarette | `modules/cigarette-status/` | Android native module. At each slot it posts a silent ongoing notification whose small icon is a cigarette, then swaps the icon every 5 minutes through 7 frames until the window closes. Runs from AlarmManager, so it works with the app closed. |
| Subscription | `src/services/subscription.ts` | Free trial for `extra.trialDays` days, then a monthly plan via RevenueCat. Falls back to a trial-only mode in development. |
| State | `src/store/useAppStore.ts` | Zustand store backed by AsyncStorage. |
| Screens | `app/` | Onboarding, Home (waiting / alert / done), Plan, Settings, Time zone picker, Paywall. |

## Get it on your phone (Android)

**Easiest, no computer tools needed: build an APK in the cloud.**

1. Create a free account at https://expo.dev and make an access token (Account settings → Access tokens).
2. In this GitHub repository open Settings → Secrets and variables → Actions and add a secret named `EXPO_TOKEN` with that token.
3. Open the Actions tab, pick "Android APK", press "Run workflow". After about 10 to 15 minutes the log ends with a link to the APK.
4. Open that link on the phone, allow installing from this source, and install. Pacer appears like any other app.

**With a computer:** install Android Studio, then

```bash
npm install
npx expo run:android --device
```

Expo Go does not include the status-bar icon module or in-app purchases, so use a real build for those. Everything else works in Expo Go.

## Quality checks

```bash
npm run typecheck   # tsc --noEmit
npm test            # jest: schedule, plan and time-zone tests
npm run lint        # eslint
```

## Set up the subscription (Google Play + RevenueCat)

1. Create a Google Play Console developer account (one-time fee) and create the app with package name `com.forroey.pacer`.
2. In Monetize → Subscriptions add a product, for example `pacer_monthly`, with a base plan at ₪10 / month. Optionally add a 1-month free offer; the app also enforces a local 30-day trial so users get the free month either way.
3. Create a free RevenueCat project, add the Android app with the Play service credentials RevenueCat asks for, and create an entitlement called `pro` attached to that product in the default offering.
4. Put the RevenueCat public Android key in `app.json` under `expo.extra.revenueCatAndroidApiKey` (iOS key goes in `revenueCatIosApiKey` when you add iPhone later).
5. Update `priceLabel` in `app.json` if the price changes. The paywall reads it.

## Ship to Google Play

```bash
npm install -g eas-cli
eas login
eas build --platform android --profile production   # produces an .aab
eas submit --platform android
```

Before submitting:

- Replace the placeholder icons in `assets/`.
- Replace the `TERMS_URL` and `PRIVACY_URL` constants in `app/paywall.tsx` with real pages. Google requires a privacy policy for apps with subscriptions.
- Fill the Data safety form: the app stores everything on device and collects no personal data. RevenueCat receives an anonymous app user id and purchase tokens.
- Describe Pacer as a habit-pacing tool, not a medical device.

### Android notes

- Notifications and the status icon need the notification permission (asked during onboarding).
- The burn-down frames use exact alarms when allowed. On Android 14+ the user can allow "Alarms & reminders" for Pacer in app settings; otherwise frames may drift by a few minutes.
- Alarms are lost on reboot; they are rescheduled the next time the app is opened.

## iPhone later

The same code builds for iOS with EAS. iOS has no app icons in the status bar, so the burning cigarette there is the one inside the app; a Live Activity on the lock screen would be the iOS equivalent and is not built yet.

## Legacy files

The repository previously held two data-analysis homework scripts and a spreadsheet. They are kept untouched under `legacy/`.
