import React, { forwardRef } from 'react';
import { Pressable, StyleSheet, Text, type PressableProps, type View } from 'react-native';
import { colors } from '../theme';

/**
 * Icon buttons drawn with text glyphs so the app has no icon-font dependency.
 * They are 44pt squares, the minimum comfortable touch target.
 */
const IconButton = forwardRef<View, PressableProps & { glyph: string; dark?: boolean }>(function IconButton(
  { glyph, dark, style, ...rest },
  ref,
) {
  return (
    <Pressable
      ref={ref}
      accessibilityRole="button"
      hitSlop={6}
      style={({ pressed }) => [styles.button, dark && styles.dark, pressed && { opacity: 0.7 }, style as object]}
      {...rest}
    >
      <Text style={[styles.glyph, dark && { color: '#FFFFFF' }]}>{glyph}</Text>
    </Pressable>
  );
});

export const GearIcon = {
  Button: forwardRef<View, PressableProps>(function Gear(props, ref) {
    return <IconButton ref={ref} glyph="⚙︎" {...props} />;
  }),
};

export const BackIcon = {
  Button: forwardRef<View, PressableProps>(function Back(props, ref) {
    return <IconButton ref={ref} glyph="‹" {...props} />;
  }),
};

export const CloseIcon = {
  Button: forwardRef<View, PressableProps>(function Close(props, ref) {
    return <IconButton ref={ref} glyph="×" {...props} />;
  }),
};

const styles = StyleSheet.create({
  button: {
    width: 44,
    height: 44,
    borderRadius: 14,
    backgroundColor: colors.surface,
    alignItems: 'center',
    justifyContent: 'center',
  },
  dark: { backgroundColor: 'rgba(255,255,255,0.18)' },
  glyph: { fontSize: 26, lineHeight: 30, fontWeight: '700', color: colors.ink },
});
