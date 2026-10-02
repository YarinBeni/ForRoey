package expo.modules.cigarettestatus

import android.content.Context
import expo.modules.kotlin.exception.Exceptions
import expo.modules.kotlin.modules.Module
import expo.modules.kotlin.modules.ModuleDefinition
import expo.modules.kotlin.records.Field
import expo.modules.kotlin.records.Record

class WindowRecord : Record {
  @Field val id: String = ""
  @Field val openAtMs: Double = 0.0
  @Field val windowMs: Double = 0.0
}

/** JavaScript-facing API; see modules/cigarette-status/index.ts. */
class CigaretteStatusModule : Module() {
  private val context: Context
    get() = appContext.reactContext ?: throw Exceptions.ReactContextLost()

  override fun definition() = ModuleDefinition {
    Name("CigaretteStatus")

    Function("scheduleWindows") { windows: List<WindowRecord> ->
      CigaretteStatus.scheduleWindows(
        context,
        windows.map { Window(it.id, it.openAtMs.toLong(), it.windowMs.toLong()) },
      )
    }

    Function("cancelAll") {
      CigaretteStatus.cancelAll(context)
    }

    Function("clear") {
      CigaretteStatus.clear(context)
    }

    /** Show the icon right now for [windowMs], for trying it out from Settings. */
    Function("preview") { windowMs: Double ->
      CigaretteStatus.showFrame(context, System.currentTimeMillis(), windowMs.toLong(), 0)
    }

    Function("canScheduleExactAlarms") {
      CigaretteStatus.canScheduleExactAlarms(context)
    }
  }
}
