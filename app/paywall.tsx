import { useRouter } from 'expo-router';
import React, { useState } from 'react';
import { Alert, Linking, Platform, Pressable, ScrollView, StyleSheet, Text, View } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { CloseIcon } from '../src/components/icons';
import { Body, Button, Chip, Title } from '../src/components/ui';
import { t } from '../src/i18n';
import { trialDaysFromConfig } from '../src/services/subscription';
import { getSubscriptionService, useAppStore } from '../src/store/useAppStore';
import { colors, spacing } from '../src/theme';

const subscriptionService = getSubscriptionService();

const TERMS_URL = 'https://example.com/pacer/terms';
const PRIVACY_URL = 'https://example.com/pacer/privacy';

export default function PaywallScreen() {
  const router = useRouter();
  const subscription = useAppStore((s) => s.subscription);
  const refreshSubscription = useAppStore((s) => s.refreshSubscription);
  const [busy, setBusy] = useState(false);

  const expired = subscription?.kind === 'expired';
  const trialDays = trialDaysFromConfig();
  const price = t('paywall.perMonth', { price: subscriptionService.priceLabel });
  const store = Platform.OS === 'ios' ? t('settings.appStore') : t('settings.playStore');
  const perks = [t('paywall.perk1'), t('paywall.perk2'), t('paywall.perk3')];

  const run = async (action: () => Promise<unknown>, successTitle: string) => {
    setBusy(true);
    try {
      await action();
      await refreshSubscription();
      const status = useAppStore.getState().subscription;
      if (status?.kind === 'active') {
        Alert.alert(successTitle, t('paywall.thanks'));
        if (router.canGoBack()) router.back();
        else router.replace('/');
      } else {
        Alert.alert(t('paywall.notActiveTitle'), t('paywall.notActiveBody'));
      }
    } catch (e) {
      Alert.alert(t('paywall.failedTitle'), e instanceof Error ? e.message : t('paywall.tryAgain'));
    } finally {
      setBusy(false);
    }
  };

  return (
    <SafeAreaView style={styles.screen}>
      <ScrollView contentContainerStyle={{ gap: 28, paddingBottom: spacing.lg }} showsVerticalScrollIndicator={false}>
        <View style={{ flexDirection: 'row', justifyContent: 'flex-end' }}>
          {!expired ? <CloseIcon.Button accessibilityLabel={t('common.close')} onPress={() => router.back()} /> : <View style={{ height: 44 }} />}
        </View>

        <View style={{ gap: 12 }}>
          {subscription?.kind === 'trial' ? (
            <Chip tone="amber">
              {subscription.daysLeft === 0 ? t('paywall.endsToday') : t('paywall.endsIn', { count: subscription.daysLeft })}
            </Chip>
          ) : expired ? (
            <Chip tone="amber">{t('paywall.ended')}</Chip>
          ) : null}
          <Title size={34}>{t('paywall.title')}</Title>
          <Body>{t('paywall.body', { days: trialDays, price, store })}</Body>
        </View>

        <View style={{ gap: 14 }}>
          {perks.map((perk) => (
            <View key={perk} style={{ flexDirection: 'row', alignItems: 'center', gap: 12 }}>
              <View style={styles.check}>
                <Text style={styles.checkGlyph}>✓</Text>
              </View>
              <Text style={styles.perk}>{perk}</Text>
            </View>
          ))}
        </View>

        <View style={styles.priceCard}>
          <View style={{ gap: 2, flex: 1 }}>
            <Text style={styles.priceTitle}>{t('paywall.monthly')}</Text>
            <Text style={styles.priceHint}>{t('paywall.billedMonthly')}</Text>
          </View>
          <Text style={styles.price}>{price}</Text>
        </View>
      </ScrollView>

      <View style={{ gap: 14, alignItems: 'center' }}>
        <Button
          title={busy ? t('paywall.oneMoment') : t('paywall.subscribeFor', { price })}
          disabled={busy}
          style={{ alignSelf: 'stretch' }}
          onPress={() => void run(() => subscriptionService.purchaseMonthly(), t('paywall.subscribedTitle'))}
        />
        <View style={{ flexDirection: 'row', gap: 20, alignItems: 'center', flexWrap: 'wrap', justifyContent: 'center' }}>
          <Button
            title={t('paywall.restore')}
            variant="ghost"
            style={{ paddingHorizontal: 0 }}
            disabled={busy}
            onPress={() => void run(() => subscriptionService.restore(), t('paywall.restoredTitle'))}
          />
          <Pressable accessibilityRole="link" hitSlop={8} onPress={() => void Linking.openURL(TERMS_URL)}>
            <Text style={styles.legal}>{t('paywall.terms')}</Text>
          </Pressable>
          <Pressable accessibilityRole="link" hitSlop={8} onPress={() => void Linking.openURL(PRIVACY_URL)}>
            <Text style={styles.legal}>{t('paywall.privacy')}</Text>
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
    gap: 12,
    padding: 18,
    borderRadius: 20,
    borderWidth: 2,
    borderColor: colors.teal,
  },
  priceTitle: { fontSize: 17, fontWeight: '800', color: colors.ink },
  priceHint: { fontSize: 14, color: colors.muted },
  price: { fontSize: 20, fontWeight: '800', letterSpacing: -0.5, color: colors.ink },
  legal: { fontSize: 14, fontWeight: '700', color: colors.muted, paddingVertical: 12 },
});
