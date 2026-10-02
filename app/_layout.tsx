import { Stack } from 'expo-router';
import { StatusBar } from 'expo-status-bar';
import React, { useEffect } from 'react';
import { ActivityIndicator, AppState, View } from 'react-native';
import { SafeAreaProvider } from 'react-native-safe-area-context';
import { configureNotificationHandler, ensureSlotChannel } from '../src/services/notifications';
import { useAppStore } from '../src/store/useAppStore';
import { colors } from '../src/theme';

configureNotificationHandler();

export default function RootLayout() {
  const hydrated = useAppStore((s) => s.hydrated);
  const onboarded = useAppStore((s) => s.onboarded);
  const language = useAppStore((s) => s.settings.language);
  const hydrate = useAppStore((s) => s.hydrate);
  const refreshToday = useAppStore((s) => s.refreshToday);
  const refreshSubscription = useAppStore((s) => s.refreshSubscription);

  useEffect(() => {
    void hydrate();
  }, [hydrate]);

  // Keep the Android notification channel named in the current language.
  useEffect(() => {
    if (hydrated) ensureSlotChannel();
  }, [hydrated, language]);

  // Whenever the app comes back to the foreground, rebuild the upcoming days
  // and re-sync the scheduled notifications so the queue never runs dry.
  useEffect(() => {
    const sub = AppState.addEventListener('change', (state) => {
      if (state === 'active' && useAppStore.getState().onboarded) {
        void refreshToday();
        void refreshSubscription();
      }
    });
    return () => sub.remove();
  }, [refreshToday, refreshSubscription]);

  if (!hydrated) {
    return (
      <View style={{ flex: 1, alignItems: 'center', justifyContent: 'center', backgroundColor: colors.ground }}>
        <ActivityIndicator color={colors.teal} />
      </View>
    );
  }

  return (
    <SafeAreaProvider>
      <StatusBar style="dark" />
      <Stack screenOptions={{ headerShown: false, contentStyle: { backgroundColor: colors.ground } }}>
        <Stack.Protected guard={!onboarded}>
          <Stack.Screen name="onboarding" />
        </Stack.Protected>
        <Stack.Protected guard={onboarded}>
          <Stack.Screen name="index" />
          <Stack.Screen name="plan" />
          <Stack.Screen name="settings" />
          <Stack.Screen name="paywall" options={{ presentation: 'modal', gestureEnabled: false }} />
        </Stack.Protected>
        <Stack.Screen name="timezone" options={{ presentation: 'modal' }} />
        <Stack.Screen name="language" options={{ presentation: 'modal' }} />
      </Stack>
    </SafeAreaProvider>
  );
}
