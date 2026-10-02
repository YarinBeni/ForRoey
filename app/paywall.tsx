import { useRouter } from 'expo-router';
import React, { useState } from 'react';
import { Alert, Linking, Pressable, ScrollView, StyleSheet, Text, View } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { CloseIcon } from '../src/components/icons';
import { Body, Button, Chip, Title } from '../src/components/ui';
import { trialDaysFromConfig } from '../src/services/subscription';
import { getSubscriptionService, useAppStore } from '../src/store/useAppStore';
import { colors, spacing } from '../src/theme';

const subscriptionService = getSubscriptionService();

const TERMS_URL = 'https://example.com/pacer/terms';
const PRIVACY_URL = 'https://example.com/pacer/privacy';

const PERKS = [
  'Random alerts, only in your awake hours',
  'A step-down plan that ends at zero',
  'Your progress, streaks and quit date',
];

export default function PaywallScreen() {
  const router = useRouter();
  const subscription = useAppStore((s) => s.subscription);
  const refreshSubscription = useAppStore((s) => s.refreshSubscription);
  const [busy, setBusy] = useState(false);

  const expired = subscription?.kind === 'expired';
  const trialDays = trialDaysFromConfig();

  const run = async (action: () => Promise<unknown>, successTitle: string) => {
    setBusy(true);
    try {
      await action();
      await refreshSubscription();
      const status = useAppStore.getState().subscription;
      if (status?.kind === 'active') {
        Alert.alert(successTitle, 'Thank you. Your alerts keep going.');
        if (router.canGoBack()) router.back();
        else router.replace('/');
      } else {
        Alert.alert('Not active yet', 'We could not find an active subscription for this Apple ID.');
      }
    } catch (e) {
      Alert.alert('Purchase not completed', e instanceof Error ? e.message : 'Please try again.');
    } finally {
      setBusy(false);
    }
  };

  return (
    <SafeAreaView style={styles.screen}>
      <ScrollView contentContainerStyle={{ gap: 28, paddingBottom: spacing.lg }} showsVerticalScrollIndicator={false}>
        <View style={{ flexDirection: 'row', justifyContent: 'flex-end' }}>
          {!expired ? <CloseIcon.Button accessibilityLabel="Close" onPress={() => router.back()} /> : <View style={{ height: 44 }} />}
        </View>

        <View style={{ gap: 12 }}>
          {subscription?.kind === 'trial' ? (
            <Chip tone="amber">
              {subscription.daysLeft === 0 ? 'Your free month ends today' : `Your free month ends in ${subscription.daysLeft} days`}
            </Chip>
          ) : expired ? (
            <Chip tone="amber">Your free month has ended</Chip>
          ) : null}
          <Title size={34}>Keep going, for less than a pack.</Title>
          <Body>
            Pacer stays free for your first {trialDays} days. After that it is {subscriptionService.priceLabel}. Cancel any
            time in the App Store.
          </Body>
        </View>

        <View style={{ gap: 14 }}>
          {PERKS.map((perk) => (
            <View key={perk} style={{ flexDirection: 'row', alignItems: 'center', gap: 12 }}>
              <View style={styles.check}>
                <Text style={styles.checkGlyph}>✓</Text>
              </View>
              <Text style={styles.perk}>{perk}</Text>
            </View>
          ))}
        </View>

        <View style={styles.priceCard}>
          <View style={{ gap: 2 }}>
            <Text style={styles.priceTitle}>Monthly</Text>
            <Text style={styles.priceHint}>Billed monthly after the free month</Text>
          </View>
          <Text style={styles.price}>{subscriptionService.priceLabel}</Text>
        </View>
      </ScrollView>

      <View style={{ gap: 14, alignItems: 'center' }}>
        <Button
          title={busy ? 'One moment…' : `Subscribe for ${subscriptionService.priceLabel}`}
          disabled={busy}
          style={{ alignSelf: 'stretch' }}
          onPress={() => void run(() => subscriptionService.purchaseMonthly(), 'Subscribed')}
        />
        <View style={{ flexDirection: 'row', gap: 20, alignItems: 'center' }}>
          <Button title="Restore purchases" variant="ghost" style={{ paddingHorizontal: 0 }} disabled={busy} onPress={() => void run(() => subscriptionService.restore(), 'Restored')} />
          <Pressable accessibilityRole="link" hitSlop={8} onPress={() => void Linking.openURL(TERMS_URL)}>
            <Text style={styles.legal}>Terms</Text>
          </Pressable>
          <Pressable accessibilityRole="link" hitSlop={8} onPress={() => void Linking.openURL(PRIVACY_URL)}>
            <Text style={styles.legal}>Privacy</Text>
          </Pressable>
        </View>
      </View>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  screen: { flex: 1, backgroundColor: colors.ground, paddingHorizontal: spacing.lg, paddingTop: spacing.md, paddingBottom: spacing.md },
  check: { width: 28, height: 28, borderRadius: 999, backgroundColor: colors.tealSoft, alignItems: 'center', justifyContent: 'center' },
  checkGlyph: { fontSize: 15, fontWeight: '800', color: colors.tealDeep },
  perk: { fontSize: 16, fontWeight: '600', color: colors.ink, flex: 1 },
  priceCard: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    padding: 18,
    borderRadius: 20,
    borderWidth: 2,
    borderColor: colors.teal,
  },
  priceTitle: { fontSize: 17, fontWeight: '800', color: colors.ink },
  priceHint: { fontSize: 14, color: colors.muted },
  price: { fontSize: 22, fontWeight: '800', letterSpacing: -0.5, color: colors.ink },
  legal: { fontSize: 14, fontWeight: '700', color: colors.muted, paddingVertical: 12 },
});
