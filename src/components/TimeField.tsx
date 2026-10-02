import DateTimePicker, { type DateTimePickerEvent } from '@react-native-community/datetimepicker';
import React, { useState } from 'react';
import { Platform, Pressable, StyleSheet, Text, View } from 'react-native';
import { formatMinuteOfDay } from '../domain/time';
import { colors, radius } from '../theme';

/**
 * Picks a wall-clock time as minutes after midnight.
 * iOS shows the native compact picker inline; Android opens the dialog on tap.
 */
export function TimeField({
  label,
  value,
  onChange,
}: {
  label: string;
  value: number;
  onChange: (minuteOfDay: number) => void;
}) {
  const [open, setOpen] = useState(false);
  const date = new Date(2000, 0, 1, Math.floor(value / 60), value % 60);

  const handle = (event: DateTimePickerEvent, picked?: Date) => {
    if (Platform.OS === 'android') setOpen(false);
    if (event.type === 'dismissed' || !picked) return;
    onChange(picked.getHours() * 60 + picked.getMinutes());
  };

  return (
    <View style={styles.field}>
      <Text style={styles.label}>{label}</Text>
      {Platform.OS === 'ios' ? (
        <DateTimePicker
          value={date}
          mode="time"
          display="compact"
          onChange={handle}
          accentColor={colors.teal}
          style={{ alignSelf: 'flex-start', marginLeft: -8 }}
        />
      ) : (
        <>
          <Pressable accessibilityRole="button" accessibilityLabel={`${label}, ${formatMinuteOfDay(value)}`} onPress={() => setOpen(true)}>
            <Text style={styles.value}>{formatMinuteOfDay(value)}</Text>
          </Pressable>
          {open ? <DateTimePicker value={date} mode="time" is24Hour display="spinner" onChange={handle} /> : null}
        </>
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  field: {
    flex: 1,
    gap: 6,
    padding: 16,
    borderRadius: radius.md,
    backgroundColor: colors.surface,
  },
  label: { fontSize: 13, fontWeight: '600', color: colors.muted },
  value: { fontSize: 28, fontWeight: '800', letterSpacing: -0.5, color: colors.ink },
});
