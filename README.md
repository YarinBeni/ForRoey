# Pacer

A small iOS app that helps you smoke less, then stop.

You tell Pacer when you are awake. Every day it picks random moments inside that window, at least 30 minutes apart, and sends you a notification: *you can have a cigarette now*. You start at 5 a day. Every week the count drops by one, until it reaches zero.

Built with Expo (React Native + TypeScript), Expo Router, local notifications, and a RevenueCat-backed monthly subscription with a free first month.

## How it works

| Piece | Where | What it does |
| --- | --- | --- |
| Schedule engine | `src/domain/schedule.ts` | Draws N random slot times inside the awake window with a minimum gap. Seeded by the plan and the date, so re-opening the app never moves today's times. |
| Step-down plan | `src/domain/plan.ts` | 5 → 4 → 3 → 2 → 1 → 0 a day, one step every `daysPerStage` days (default 7). Computes the quit date. |
| Time zones | `src/domain/time.ts` | Wall-clock ↔ instant conversion with `Intl` only, DST-safe. |
| Notifications | `src/services/notifications.ts` | Schedules the next ~7 days of slots as local notifications (iOS allows 64 pending). Re-synced every time the app comes to the foreground. |
| Subscription | `src/services/subscription.ts` | Free trial for `extra.trialDays` days, then a monthly plan via RevenueCat. Falls back to a trial-only mode in development. |
| State | `src/store/useAppStore.ts` | Zustand store backed by AsyncStorage. |
| Screens | `app/` | Onboarding, Home (waiting / alert / done), Plan, Settings, Time zone picker, Paywall. |

## Run it

```bash
npm install
npm run ios        # needs Xcode + a simulator, or scan the QR with Expo Go
```

Notifications do not fire on the iOS simulator. For a real test, run on a device:

```bash
npx expo run:ios --device
```

Expo Go works for everything except the in-app purchase. The app detects Expo Go and uses the trial-only subscription mode there.

## Quality checks

```bash
npm run typecheck   # tsc --noEmit
npm test            # jest: schedule, plan and time-zone tests
npm run lint        # eslint
```

## Set up the subscription (RevenueCat)

1. Create the app in App Store Connect with bundle id `com.forroey.pacer`.
2. Add an auto-renewable subscription, for example `pacer_monthly`, priced at ₪10 / month. Add a 1-month free introductory offer if you want Apple to run the trial; the app also enforces a local 30-day trial so users get the free month either way.
3. Create a RevenueCat project, add the iOS app, paste the App Store shared secret, and create an entitlement called `pro` attached to that product in the default offering.
4. Put the RevenueCat public iOS key in `app.json` under `expo.extra.revenueCatIosApiKey`.
5. Update `priceLabel` in `app.json` if the price changes. The paywall reads it.

## Ship to the App Store

```bash
npm install -g eas-cli
eas login
eas build:configure
eas build --platform ios --profile production
eas submit --platform ios
```

Before submitting:

- Replace the placeholder icons in `assets/`.
- Replace the `TERMS_URL` and `PRIVACY_URL` constants in `app/paywall.tsx` with real pages. Apple requires both for subscriptions.
- Fill the App Privacy questionnaire: the app stores everything on device and collects no personal data. RevenueCat receives an anonymous app user id and purchase receipts.
- Health note for review: Pacer is a habit-pacing tool, not a medical device. The description should say so.

## Legacy files

The repository previously held two data-analysis homework scripts and a spreadsheet. They are kept untouched under `legacy/`.
