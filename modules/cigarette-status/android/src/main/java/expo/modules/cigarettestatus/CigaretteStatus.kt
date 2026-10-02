package expo.modules.cigarettestatus

import android.app.AlarmManager
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.net.Uri
import android.os.Build
import androidx.core.app.NotificationCompat
import androidx.core.app.NotificationManagerCompat
import androidx.core.content.ContextCompat

/** One upcoming smoking window: when it opens and how long it lasts. */
data class Window(val id: String, val openAtMs: Long, val windowMs: Long)

/**
 * Shows a burning cigarette in the Android status bar while a smoking window
 * is open.
 *
 * At the window's start an alarm posts a silent, ongoing notification whose
 * small icon is a whole cigarette. Follow-up alarms swap the icon for shorter
 * and shorter frames until the window closes, when the cigarette is burnt down
 * to the filter. Everything runs natively from AlarmManager, so it works while
 * the app is in the background or closed.
 */
object CigaretteStatus {
  const val CHANNEL_ID = "pacer_smoking_window"
  const val NOTIFICATION_ID = 0x5157
  const val ACTION_OPEN = "expo.modules.cigarettestatus.action.OPEN"
  const val ACTION_FRAME = "expo.modules.cigarettestatus.action.FRAME"
  const val EXTRA_OPEN_AT = "openAtMs"
  const val EXTRA_WINDOW = "windowMs"
  const val EXTRA_FRAME = "frame"

  /** Frame 0 is a whole cigarette, frame LAST_FRAME is burnt out. */
  const val LAST_FRAME = 6

  private const val PREFS = "expo.modules.cigarettestatus"
  private const val KEY_IDS = "scheduledIds"
  private const val KEY_TITLE_OPEN = "titleOpen"
  private const val KEY_TITLE_CLOSED = "titleClosed"
  private const val KEY_TEXT_OPEN = "textOpen"
  private const val KEY_TEXT_CLOSED = "textClosed"
  private const val FRAME_REQUEST_CODE = 0x5158
  private const val TEAL = 0xFF157A6E.toInt()

  private val FRAME_ICONS = intArrayOf(
    R.drawable.ic_cig_0,
    R.drawable.ic_cig_1,
    R.drawable.ic_cig_2,
    R.drawable.ic_cig_3,
    R.drawable.ic_cig_4,
    R.drawable.ic_cig_5,
    R.drawable.ic_cig_6,
  )

  /** Text shown in the notification, in the app's current language. Falls back to the string resources. */
  fun setStrings(context: Context, titleOpen: String, titleClosed: String, textOpen: String, textClosed: String) {
    prefs(context).edit()
      .putString(KEY_TITLE_OPEN, titleOpen)
      .putString(KEY_TITLE_CLOSED, titleClosed)
      .putString(KEY_TEXT_OPEN, textOpen)
      .putString(KEY_TEXT_CLOSED, textClosed)
      .apply()
  }

  private fun text(context: Context, key: String, fallback: Int): String =
    prefs(context).getString(key, null) ?: context.getString(fallback)

  fun ensureChannel(context: Context) {
    if (Build.VERSION.SDK_INT < Build.VERSION_CODES.O) return
    val channel = NotificationChannel(CHANNEL_ID, context.getString(R.string.cig_channel_name), NotificationManager.IMPORTANCE_LOW).apply {
      description = context.getString(R.string.cig_channel_desc)
      setShowBadge(false)
      enableVibration(false)
    }
    val manager = context.getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager
    manager.createNotificationChannel(channel)
  }

  /** Replace every scheduled window with [windows]. Past `openAtMs` values fire right away. */
  fun scheduleWindows(context: Context, windows: List<Window>) {
    cancelAll(context)
    ensureChannel(context)
    val alarmManager = context.getSystemService(Context.ALARM_SERVICE) as AlarmManager
    val ids = HashSet<String>()
    for (window in windows) {
      if (window.windowMs <= 0L) continue
      setAlarm(alarmManager, window.openAtMs, openIntent(context, window))
      ids.add(window.id)
    }
    prefs(context).edit().putStringSet(KEY_IDS, ids).apply()
  }

  /** Cancel every pending window alarm and hide the current status icon. */
  fun cancelAll(context: Context) {
    val alarmManager = context.getSystemService(Context.ALARM_SERVICE) as AlarmManager
    val ids = prefs(context).getStringSet(KEY_IDS, emptySet()) ?: emptySet()
    for (id in ids) {
      alarmManager.cancel(openIntent(context, Window(id, 0L, 0L)))
    }
    prefs(context).edit().remove(KEY_IDS).apply()
    clear(context)
  }

  /** Hide the status icon and stop the burn-down of the window that is open now. */
  fun clear(context: Context) {
    val alarmManager = context.getSystemService(Context.ALARM_SERVICE) as AlarmManager
    alarmManager.cancel(frameIntent(context, 0L, 0L, 0))
    NotificationManagerCompat.from(context).cancel(NOTIFICATION_ID)
  }

  /** Post frame [frame] of the window and queue the next one. */
  fun showFrame(context: Context, openAtMs: Long, windowMs: Long, frame: Int) {
    ensureChannel(context)
    val f = frame.coerceIn(0, LAST_FRAME)
    val finished = f >= LAST_FRAME
    val endMs = openAtMs + windowMs

    val launch = context.packageManager.getLaunchIntentForPackage(context.packageName)
    val contentIntent = launch?.let { PendingIntent.getActivity(context, 0, it, pendingFlags()) }

    val builder = NotificationCompat.Builder(context, CHANNEL_ID)
      .setSmallIcon(FRAME_ICONS[f])
      .setColor(TEAL)
      .setContentTitle(
        if (finished) text(context, KEY_TITLE_CLOSED, R.string.cig_title_closed)
        else text(context, KEY_TITLE_OPEN, R.string.cig_title_open),
      )
      .setContentText(
        if (finished) text(context, KEY_TEXT_CLOSED, R.string.cig_text_closed)
        else text(context, KEY_TEXT_OPEN, R.string.cig_text_open),
      )
      .setOngoing(!finished)
      .setOnlyAlertOnce(true)
      .setSilent(true)
      .setCategory(NotificationCompat.CATEGORY_STATUS)
      .setProgress(LAST_FRAME, f, false)
      .setContentIntent(contentIntent)

    if (finished) {
      builder.setAutoCancel(true).setTimeoutAfter(10L * 60L * 1000L)
    } else {
      // A live countdown to the end of the window, rendered by the system.
      builder.setWhen(endMs).setShowWhen(true).setUsesChronometer(true).setChronometerCountDown(true)
    }

    if (canPostNotifications(context)) {
      NotificationManagerCompat.from(context).notify(NOTIFICATION_ID, builder.build())
    }

    if (!finished) {
      val nextFrame = f + 1
      val at = openAtMs + (windowMs * nextFrame) / LAST_FRAME
      val alarmManager = context.getSystemService(Context.ALARM_SERVICE) as AlarmManager
      setAlarm(alarmManager, at, frameIntent(context, openAtMs, windowMs, nextFrame))
    }
  }

  fun canScheduleExactAlarms(context: Context): Boolean {
    if (Build.VERSION.SDK_INT < Build.VERSION_CODES.S) return true
    val alarmManager = context.getSystemService(Context.ALARM_SERVICE) as AlarmManager
    return alarmManager.canScheduleExactAlarms()
  }

  // ---------------------------------------------------------------------------

  private fun canPostNotifications(context: Context): Boolean {
    if (Build.VERSION.SDK_INT < 33) return NotificationManagerCompat.from(context).areNotificationsEnabled()
    return ContextCompat.checkSelfPermission(context, "android.permission.POST_NOTIFICATIONS") == PackageManager.PERMISSION_GRANTED
  }

  private fun setAlarm(alarmManager: AlarmManager, atMs: Long, operation: PendingIntent) {
    val exact = Build.VERSION.SDK_INT < Build.VERSION_CODES.S || alarmManager.canScheduleExactAlarms()
    if (exact) {
      alarmManager.setExactAndAllowWhileIdle(AlarmManager.RTC_WAKEUP, atMs, operation)
    } else {
      alarmManager.setAndAllowWhileIdle(AlarmManager.RTC_WAKEUP, atMs, operation)
    }
  }

  private fun pendingFlags(): Int = PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE

  /** The alarm that opens a window. Distinct per id via the data URI and request code. */
  private fun openIntent(context: Context, window: Window): PendingIntent {
    val intent = Intent(context, CigaretteStatusReceiver::class.java).apply {
      action = ACTION_OPEN
      data = Uri.parse("cigarette-status://window/" + Uri.encode(window.id))
      putExtra(EXTRA_OPEN_AT, window.openAtMs)
      putExtra(EXTRA_WINDOW, window.windowMs)
    }
    return PendingIntent.getBroadcast(context, window.id.hashCode(), intent, pendingFlags())
  }

  /** The alarm that advances the burn-down. Only one window is open at a time, so one request code. */
  private fun frameIntent(context: Context, openAtMs: Long, windowMs: Long, frame: Int): PendingIntent {
    val intent = Intent(context, CigaretteStatusReceiver::class.java).apply {
      action = ACTION_FRAME
      putExtra(EXTRA_OPEN_AT, openAtMs)
      putExtra(EXTRA_WINDOW, windowMs)
      putExtra(EXTRA_FRAME, frame)
    }
    return PendingIntent.getBroadcast(context, FRAME_REQUEST_CODE, intent, pendingFlags())
  }

  private fun prefs(context: Context) = context.getSharedPreferences(PREFS, Context.MODE_PRIVATE)
}
