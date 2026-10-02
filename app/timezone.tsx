import { useRouter } from 'expo-router';
import React, { useMemo, useState } from 'react';
import { FlatList, Pressable, StyleSheet, Text, TextInput, View } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { CloseIcon } from '../src/components/icons';
import { Title } from '../src/components/ui';
import { tzOffsetLabel } from '../src/domain/format';
import { t } from '../src/i18n';
import { useAppStore } from '../src/store/useAppStore';
import { colors, spacing } from '../src/theme';

const FALLBACK_ZONES = [
  'Asia/Jerusalem',
  'Europe/London',
  'Europe/Paris',
  'Europe/Berlin',
  'Europe/Athens',
  'Europe/Moscow',
  'America/New_York',
  'America/Chicago',
  'America/Denver',
  'America/Los_Angeles',
  'America/Sao_Paulo',
  'Asia/Dubai',
  'Asia/Kolkata',
  'Asia/Bangkok',
  'Asia/Shanghai',
  'Asia/Tokyo',
  'Australia/Sydney',
  'Pacific/Auckland',
];

function allZones(): string[] {
  const intl = Intl as unknown as { supportedValuesOf?: (key: string) => string[] };
  try {
    const list = intl.supportedValuesOf?.('timeZone');
    if (list && list.length > 0) return list;
  } catch {
    // Older JS engines: fall back to a curated list.
  }
  return FALLBACK_ZONES;
}

export default function TimeZoneScreen() {
  const router = useRouter();
  const onboarded = useAppStore((s) => s.onboarded);
  const current = useAppStore((s) => s.settings?.timeZone);
  const setDraft = useAppStore((s) => s.setDraftSettings);
  const updateSettings = useAppStore((s) => s.updateSettings);
  const [query, setQuery] = useState('');

  const zones = useMemo(() => {
    const q = query.trim().toLowerCase().replace(/\s+/g, '_');
    const list = allZones();
    return q ? list.filter((z) => z.toLowerCase().includes(q)) : list;
  }, [query]);

  const pick = (timeZone: string) => {
    if (onboarded) void updateSettings({ timeZone });
    else setDraft({ timeZone });
    router.back();
  };

  return (
    <SafeAreaView style={styles.screen} edges={['top', 'left', 'right']}>
      <View style={styles.header}>
        <Title size={28}>{t('timezone.title')}</Title>
        <CloseIcon.Button accessibilityLabel={t('common.close')} onPress={() => router.back()} />
      </View>
      <TextInput
        value={query}
        onChangeText={setQuery}
        placeholder={t('timezone.search')}
        placeholderTextColor={colors.muted}
        autoCapitalize="none"
        autoCorrect={false}
        style={styles.search}
        accessibilityLabel={t('timezone.search')}
      />
      <FlatList
        data={zones}
        keyExtractor={(z) => z}
        keyboardShouldPersistTaps="handled"
        renderItem={({ item }) => {
          const selected = item === current;
          return (
            <Pressable
              accessibilityRole="button"
              accessibilityState={{ selected }}
              onPress={() => pick(item)}
              style={({ pressed }) => [styles.row, pressed && { opacity: 0.7 }]}
            >
              <Text style={[styles.zone, selected && { color: colors.tealDeep, fontWeight: '800' }]}>{item.replace(/_/g, ' ')}</Text>
              <Text style={styles.offset}>{tzOffsetLabel(item)}</Text>
            </Pressable>
          );
        }}
        ItemSeparatorComponent={() => <View style={{ height: 1, backgroundColor: colors.line }} />}
      />
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  screen: { flex: 1, backgroundColor: colors.ground, paddingHorizontal: spacing.lg, paddingTop: spacing.md },
  header: { flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between', marginBottom: 12 },
  search: {
    height: 48,
    borderRadius: 14,
    backgroundColor: colors.surface,
    paddingHorizontal: 16,
    fontSize: 16,
    color: colors.ink,
    marginBottom: 8,
  },
  row: { flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between', minHeight: 52, paddingVertical: 8 },
  zone: { fontSize: 16, fontWeight: '600', color: colors.ink, flex: 1 },
  offset: { fontSize: 14, fontWeight: '600', color: colors.muted },
});
