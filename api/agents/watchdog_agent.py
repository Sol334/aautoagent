import json


class WatchdogAgent:
    name = "watchdog"
    description = "Congressional trade monitor — fetches STOCK Act disclosures and sends Telegram alerts"
    skills = ["congressional_trades", "political_watchdog", "telegram_alert"]

    def handle(self, text: str) -> str:
        try:
            text_lower = text.lower()
            test_mode = "test" in text_lower
            # Default to dry_run=True for safety; only disable when explicitly asked for live run
            dry_run = "live" not in text_lower

            from scripts.political_watchdog import run
            result = run(dry_run=dry_run, test_mode=test_mode)
            return json.dumps(result)
        except Exception as exc:
            return json.dumps({"error": str(exc)})
