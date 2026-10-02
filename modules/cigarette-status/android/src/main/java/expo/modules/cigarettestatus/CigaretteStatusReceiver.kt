package expo.modules.cigarettestatus

import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent

/** Receives the window-open and frame alarms and posts the matching status icon. */
class CigaretteStatusReceiver : BroadcastReceiver() {
  override fun onReceive(context: Context, intent: Intent) {
    val openAtMs = intent.getLongExtra(CigaretteStatus.EXTRA_OPEN_AT, 0L)
    val windowMs = intent.getLongExtra(CigaretteStatus.EXTRA_WINDOW, 0L)
    if (openAtMs <= 0L || windowMs <= 0L) return
    when (intent.action) {
      CigaretteStatus.ACTION_OPEN -> CigaretteStatus.showFrame(context, openAtMs, windowMs, 0)
      CigaretteStatus.ACTION_FRAME -> {
        val frame = intent.getIntExtra(CigaretteStatus.EXTRA_FRAME, CigaretteStatus.LAST_FRAME)
        CigaretteStatus.showFrame(context, openAtMs, windowMs, frame)
      }
    }
  }
}
