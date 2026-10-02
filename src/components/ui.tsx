import React from 'react';
import {
  Pressable,
  StyleSheet,
  Text,
  View,
  type PressableProps,
  type StyleProp,
  type TextStyle,
  type ViewStyle,
} from 'react-native';
import { colors, radius, spacing } from '../theme';

/** Small, shared building blocks that mirror the mockups. */

type ButtonProps = PressableProps & {
  title: string;
  variant?: 'primary' | 'secondary' | 'ghost' | 'danger' | 'onTeal' | 'onTealOutline';
  style?: StyleProp<ViewStyle>;
};

export function Button({ title, variant = 'primary', style, disabled, ...rest }: ButtonProps) {
  const base: ViewStyle[] = [styles.button];
  const label: TextStyle[] = [styles.buttonLabel];
  switch (variant) {
    case 'primary':
      base.push({ backgroundColor: colors.teal });
      label.push({ color: '#FFFFFF' });
      break;
    case 'secondary':
      base.push({ borderWidth: 1, borderColor: colors.line, backgroundColor: colors.ground });
      label.push({ color: colors.ink });
      break;
    case 'ghost':
      base.push({ backgroundColor: 'transparent', height: 44 });
      label.push({ color: colors.teal, fontSize: 15 });
      break;
    case 'danger':
      base.push({ borderWidth: 1, borderColor: colors.line, backgroundColor: colors.ground });
      label.push({ color: colors.danger });
      break;
    case 'onTeal':
      base.push({ backgroundColor: '#FFFFFF', height: 60, borderRadius: 18 });
      label.push({ color: colors.tealDeep, fontSize: 18, fontWeight: '800' });
      break;
    case 'onTealOutline':
      base.push({ borderWidth: 2, borderColor: 'rgba(255,255,255,0.6)', borderRadius: 18 });
      label.push({ color: '#FFFFFF' });
      break;
  }
  return (
    <Pressable
      accessibilityRole="button"
      disabled={disabled}
      style={({ pressed }) => [base, pressed && { opacity: 0.85 }, disabled && { opacity: 0.4 }, style]}
      {...rest}
    >
      <Text style={label}>{title}</Text>
    </Pressable>
  );
}

export function Eyebrow({ children, light }: { children: React.ReactNode; light?: boolean }) {
  return <Text style={[styles.eyebrow, light && { color: colors.tealLight }]}>{children}</Text>;
}

export function Title({ children, size = 32 }: { children: React.ReactNode; size?: number }) {
  return <Text style={[styles.title, { fontSize: size }]}>{children}</Text>;
}

export function Body({ children, style }: { children: React.ReactNode; style?: StyleProp<TextStyle> }) {
  return <Text style={[styles.body, style]}>{children}</Text>;
}

export function Card({ children, style }: { children: React.ReactNode; style?: StyleProp<ViewStyle> }) {
  return <View style={[styles.card, style]}>{children}</View>;
}

/** A row inside a Card: label on the left, value on the right. */
export function Row({
  label,
  hint,
  right,
  onPress,
  last,
}: {
  label: string;
  hint?: string;
  right: React.ReactNode;
  onPress?: () => void;
  last?: boolean;
}) {
  const content = (
    <>
      <View style={{ flex: 1, gap: 2 }}>
        <Text style={styles.rowLabel}>{label}</Text>
        {hint ? <Text style={styles.rowHint}>{hint}</Text> : null}
      </View>
      {typeof right === 'string' ? <Text style={styles.rowValue}>{right}</Text> : right}
    </>
  );
  const rowStyle = [styles.row, !last && styles.rowDivider];
  if (onPress) {
    return (
      <Pressable accessibilityRole="button" onPress={onPress} style={({ pressed }) => [rowStyle, pressed && { opacity: 0.7 }]}>
        {content}
      </Pressable>
    );
  }
  return <View style={rowStyle}>{content}</View>;
}

export function Stepper({
  value,
  onChange,
  min,
  max,
  step = 1,
  format = (v: number) => String(v),
  labelLess,
  labelMore,
}: {
  value: number;
  onChange: (v: number) => void;
  min: number;
  max: number;
  step?: number;
  format?: (v: number) => string;
  labelLess: string;
  labelMore: string;
}) {
  return (
    <View style={styles.stepper}>
      <Pressable
        accessibilityRole="button"
        accessibilityLabel={labelLess}
        disabled={value - step < min}
        onPress={() => onChange(Math.max(min, value - step))}
        style={({ pressed }) => [styles.stepperButton, pressed && { opacity: 0.7 }, value - step < min && { opacity: 0.35 }]}
      >
        <Text style={styles.stepperGlyph}>−</Text>
      </Pressable>
      <Text style={styles.stepperValue}>{format(value)}</Text>
      <Pressable
        accessibilityRole="button"
        accessibilityLabel={labelMore}
        disabled={value + step > max}
        onPress={() => onChange(Math.min(max, value + step))}
        style={({ pressed }) => [styles.stepperButton, pressed && { opacity: 0.7 }, value + step > max && { opacity: 0.35 }]}
      >
        <Text style={styles.stepperGlyph}>+</Text>
      </Pressable>
    </View>
  );
}

export function Chip({ children, tone = 'teal' }: { children: React.ReactNode; tone?: 'teal' | 'amber' }) {
  const bg = tone === 'teal' ? colors.tealSoft : colors.amberSoft;
  const fg = tone === 'teal' ? colors.tealDeep : colors.amberDeep;
  return (
    <View style={[styles.chip, { backgroundColor: bg }]}>
      <Text style={[styles.chipText, { color: fg }]}>{children}</Text>
    </View>
  );
}

/** The row of dots showing today's slots. */
export function Dots({
  total,
  done,
  nextIndex,
  onDark,
}: {
  total: number;
  done: boolean[];
  nextIndex: number | null;
  onDark?: boolean;
}) {
  const filled = onDark ? '#FFFFFF' : colors.teal;
  const empty = onDark ? 'rgba(255,255,255,0.3)' : colors.line;
  const ring = onDark ? '#FFFFFF' : colors.amber;
  return (
    <View style={{ flexDirection: 'row', gap: 14 }}>
      {Array.from({ length: total }).map((_, i) => {
        const isNext = i === nextIndex;
        return (
          <View
            key={i}
            style={{
              width: 22,
              height: 22,
              borderRadius: 999,
              backgroundColor: done[i] ? filled : isNext ? 'transparent' : empty,
              borderWidth: isNext ? 3 : 0,
              borderColor: ring,
            }}
          />
        );
      })}
    </View>
  );
}

export function SectionLabel({ children }: { children: React.ReactNode }) {
  return <Text style={styles.eyebrow}>{children}</Text>;
}

const styles = StyleSheet.create({
  button: {
    height: 56,
    borderRadius: radius.md,
    alignItems: 'center',
    justifyContent: 'center',
    paddingHorizontal: spacing.md,
  },
  buttonLabel: { fontSize: 17, fontWeight: '700' },
  eyebrow: {
    fontSize: 13,
    fontWeight: '700',
    letterSpacing: 1,
    textTransform: 'uppercase',
    color: colors.muted,
  },
  title: { fontWeight: '800', letterSpacing: -0.5, color: colors.ink, lineHeight: 36 },
  body: { fontSize: 16, lineHeight: 23, color: colors.muted },
  card: {
    borderWidth: 1,
    borderColor: colors.line,
    borderRadius: radius.md,
    overflow: 'hidden',
    backgroundColor: colors.ground,
  },
  row: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingVertical: 12,
    paddingHorizontal: spacing.md,
    minHeight: 56,
    gap: spacing.sm,
  },
  rowDivider: { borderBottomWidth: 1, borderBottomColor: colors.line },
  rowLabel: { fontSize: 16, fontWeight: '600', color: colors.ink },
  rowHint: { fontSize: 13, color: colors.muted },
  rowValue: { fontSize: 16, fontWeight: '800', color: colors.ink },
  stepper: { flexDirection: 'row', alignItems: 'center', gap: 8 },
  stepperButton: {
    width: 44,
    height: 44,
    borderRadius: 12,
    backgroundColor: colors.surface,
    alignItems: 'center',
    justifyContent: 'center',
  },
  stepperGlyph: { fontSize: 22, fontWeight: '700', color: colors.ink, lineHeight: 26 },
  stepperValue: { minWidth: 56, textAlign: 'center', fontSize: 18, fontWeight: '800', color: colors.ink },
  chip: { alignSelf: 'flex-start', paddingVertical: 6, paddingHorizontal: 12, borderRadius: 999 },
  chipText: { fontSize: 13, fontWeight: '700' },
});
