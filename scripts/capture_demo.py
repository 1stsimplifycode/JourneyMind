"""Drive the running app over localhost and capture the demo, step by step.

    python scripts/capture_demo.py            # needs the server on :8000

This is loopback verification, not a screenshot tour: every step asserts that
the thing it is about to photograph is actually on screen. If the reveal panel
never appears, or the fare does not move between attempts, the script fails
with the reason rather than saving a picture of a broken page.

It drives the Chrome already installed on the machine (`channel="chrome"`), so
no separate browser download is needed.
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "screenshot"
BASE = "http://127.0.0.1:8000"
VIEWPORT = {"width": 1440, "height": 960}


NL = chr(10)


class StepFailed(RuntimeError):
    pass


def money(text: str) -> float | None:
    m = re.search(r"₹\s*([\d,]+)", text or "")
    return float(m.group(1).replace(",", "")) if m else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=BASE)
    ap.add_argument("--headed", action="store_true", help="watch it run")
    ap.add_argument("--keep", action="store_true", help="do not clear screenshot/")
    args = ap.parse_args()

    from playwright.sync_api import TimeoutError as PWTimeout
    from playwright.sync_api import sync_playwright

    if OUT.exists() and not args.keep:
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True, exist_ok=True)

    shots: list[tuple[str, str]] = []
    findings: list[str] = []

    def shot(page, name: str, caption: str) -> None:
        """Number the step here, so inserting one does not renumber the rest."""
        n = len(shots) + 1
        path = OUT / f"{n:02d}-{name}.png"
        page.screenshot(path=str(path), full_page=False)
        shots.append((path.name, caption))
        print(f"  [{n:02d}] {path.name:34s} {caption}")

    with sync_playwright() as pw:
        browser = pw.chromium.launch(channel="chrome", headless=not args.headed)
        page = browser.new_page(viewport=VIEWPORT, device_scale_factor=2)
        page.set_default_timeout(20_000)

        # A React view that throws unmounts silently and the next wait_for
        # times out forty seconds later with no clue why. Collect the actual
        # error instead. Publishing `expected: null` for unroutable options
        # crashed the whole Intelligence view this way.
        console_errors: list[str] = []
        failed_requests: list[str] = []
        page.on("console", lambda m: console_errors.append(m.text)
                if m.type == "error" else None)
        page.on("pageerror", lambda e: console_errors.append(f"uncaught: {e}"))
        page.on("requestfailed", lambda r: failed_requests.append(
            f"{r.method} {r.url} — {r.failure}"))
        page.on("response", lambda r: failed_requests.append(
            f"HTTP {r.status} {r.url}") if r.status >= 400 else None)

        # -- 1. the product, with no model in sight ------------------------
        page.goto(args.base, wait_until="networkidle")
        page.wait_for_selector(".ridecard", timeout=30_000)
        cards = page.locator(".ridelist").first.locator(".ridecard:not(.out):not(.journeycard)")
        n = cards.count()
        if n < 3:
            raise StepFailed(f"only {n} bookable options rendered")

        # The front door must read as a ride app. Model vocabulary here would
        # answer the question before the rider has felt it.
        body = page.inner_text("body")
        for banned in ("expected cost", "Expected cost", "GNN", "GraphSAGE",
                       "GAT", "probability", "prototype", "Markov"):
            if banned in body:
                raise StepFailed(
                    f"the booking screen mentions {banned!r} — the reveal is "
                    f"supposed to come after the failure, not before it")
        # You do not book a metro; you turn up. A BOOK NOW on a timetabled
        # service claims a ticketing integration that does not exist.
        for i in range(n):
            card = cards.nth(i)
            label = card.locator("button.booknow, button.viewjourney").first.inner_text()
            name = card.locator("h3").inner_text()
            if any(w in name for w in ("Metro", "Bus")):
                if "Book now" in label:
                    raise StepFailed(f"{name.strip()} offers BOOK NOW with no "
                                     f"ticketing integration behind it")

        shot(page, "book-view", "looks like a normal ride app")

        # Cards must be cheapest-first: the whole demo turns on the rider's eye
        # landing on the cheap option.
        fares = []
        for i in range(n):
            c = cards.nth(i)
            fares.append((money(c.locator(".ridecard-fare").inner_text()),
                          c.locator("h3").inner_text(), i))
        fares = [f for f in fares if f[0] is not None]
        if fares != sorted(fares):
            raise StepFailed(
                "bookable options are not sorted cheapest-first: "
                + ", ".join(f"{nm} ₹{fa:.0f}" for fa, nm, _ in fares))
        cheapest_fare, cheapest_name, cheapest_i = fares[0]
        print(f"       cheapest on screen: {cheapest_name} at ₹{cheapest_fare:.0f} (first card)")

        # The card the demo presses is the cheapest HAILED ride, not simply the
        # cheapest card. On this corridor the cheapest option is a bus, which
        # completes -- and a rider due at a meeting in an hour does not take a
        # ninety-minute bus. Booking it would also mean the story this product
        # exists to tell (a driver accepts, then cancels) never happens, because
        # a timetable has no driver to cancel.
        book_i, book_name, book_fare = None, None, None
        for i in range(n):
            card = cards.nth(i)
            if card.locator("button.booknow", has_text="Book now").count():
                book_i = i
                book_name = card.locator("h3").inner_text().split("·")[0].strip()
                book_fare = money(card.locator(".ridecard-fare").inner_text())
                break
        if book_i is None:
            raise StepFailed("no hailed ride on screen to book")
        print(f"       booking: {book_name} at ₹{book_fare:.0f} "
              f"(cheapest ride you can hail)")

        # -- 1b. the planner is on the booking screen, not hidden in a tab --
        journeys = page.locator(".journeycard")
        jn = journeys.count()
        if jn == 0:
            raise StepFailed(
                "no multi-stage journeys on the booking screen — the product is "
                "back to offering single vehicles only")
        # A "journey" with one vehicle in it is just a ride with extra words.
        for i in range(jn):
            legs = journeys.nth(i).locator(".journeydots i").count()
            if legs < 2:
                raise StepFailed(f"journey {i + 1} has {legs} legs")
        # Walking is not bookable, so a journey card must not offer BOOK NOW.
        if journeys.locator("button.booknow").count():
            raise StepFailed("a journey card offers BOOK NOW — you book the "
                             "rides inside a journey, not the journey")
        journeys.first.scroll_into_view_if_needed()
        page.wait_for_timeout(300)
        shot(page, "journeys", f"{jn} multi-stage journeys, no BOOK NOW")

        journeys.first.locator("button.viewjourney").click()
        page.wait_for_selector(".journeylegs li", timeout=8_000)
        page.wait_for_timeout(400)
        leg_count = journeys.first.locator(".journeylegs li").count()
        print(f"       journey expands to {leg_count} legs")
        shot(page, "journey-legs", "every leg, stop by stop")
        journeys.first.locator("button.viewjourney").click()   # tidy up
        page.wait_for_timeout(250)

        # -- 2. press BOOK NOW and watch it play out -----------------------
        # The cheapest card is what a rider's eye lands on, and in the primary
        # list it is a hailed ride -- the one that can actually be cancelled.
        cards.nth(book_i).locator("button.booknow").click()
        page.wait_for_selector(".bookpanel", timeout=15_000)
        page.wait_for_selector(".steps .step", timeout=15_000)
        page.wait_for_timeout(700)
        shot(page, "booking-searching", "searching for a driver")

        def settle() -> None:
            """Wait for the attempt animation to finish."""
            page.wait_for_selector(".bookdone", timeout=25_000)

        settle()
        outcome_1 = page.inner_text(".bookdone")
        failed_1 = "could not be completed" in outcome_1
        shot(page, "booking-outcome-1",
             "first attempt " + ("fails" if failed_1 else "succeeds"))
        if not failed_1:
            findings.append(
                "NOTE: the first attempt succeeded, so the failure path was not "
                "exercised. Demo mode should normally fail this option first.")

        # -- 3. try again: the fare moves ----------------------------------
        retried = False
        if page.locator("button.booknow", has_text="Try again").count():
            first_fare = money(page.inner_text(".bookpanel-head h3"))
            page.locator("button.booknow", has_text="Try again").first.click()
            page.wait_for_timeout(600)
            settle()
            retried = True
            second_fare = money(page.inner_text(".bookpanel-head h3"))
            if first_fare and second_fare and second_fare <= first_fare:
                raise StepFailed(
                    f"retry did not reprice: ₹{first_fare:.0f} -> ₹{second_fare:.0f}")
            print(f"       retry repriced: ₹{first_fare:.0f} -> ₹{second_fare:.0f}")
            shot(page, "booking-retry",
                 f"retry at a higher fare (₹{first_fare:.0f} → ₹{second_fare:.0f})")

        # -- 3b. spend the retry budget: what a consumer app never shows ---
        spent = 1 + (1 if retried else 0)
        while page.locator("button.booknow", has_text="Try again").count():
            page.locator("button.booknow", has_text="Try again").first.click()
            page.wait_for_timeout(500)
            settle()
            spent += 1
            if spent > 8:
                raise StepFailed("the retry button never went away")
        print(f"       retry budget spent after {spent} attempts")

        # An arrival-risk panel under the words "Journey completed" is the
        # product contradicting itself: the rider is in the vehicle. Only a
        # rider who never got a ride is stranded.
        settled = "Journey completed" in page.inner_text(".bookpanel")
        esc = page.locator(".escbox")
        if settled and esc.count():
            raise StepFailed(
                "the booking completed and the escalation still fired — "
                "'you may be late, switch to Cab' under 'Journey completed'")
        if not settled and not esc.count():
            findings.append(
                "NOTE: the retry budget was spent with no ride and no "
                "escalation appeared")

        if esc.count():
            esc.scroll_into_view_if_needed()
            page.wait_for_timeout(500)
            head = esc.locator("h3").inner_text()
            print(f"       escalation: {head}")
            # A projection that puts the rider hours early is the wall-clock bug.
            detail = esc.inner_text()
            # Mode keys are for joins. "Switch to bike_taxi then bus then
            # metro" is a database row read aloud to a rider.
            for key in ("bike_taxi", "namma_yatri", "_taxi"):
                if key in detail:
                    raise StepFailed(f"raw mode key {key!r} in the escalation panel")
            if "completes 0% of the time" in detail:
                raise StepFailed("an option reported as completing 0% of the time")
            if re.search(r"[0-9]{3,} minutes", detail):
                raise StepFailed(f"implausible arrival projection: {detail[:160]}")
            shot(page, "escalation",
                 "after four failures, the meeting is at risk")

            notify = esc.locator("button.booknow", has_text="Notify manager")
            if notify.count():
                notify.click()
                page.wait_for_selector(".escsent pre", timeout=15_000)
                page.wait_for_timeout(500)
                msg = page.inner_text(".escsent")
                if "composed_not_sent" not in msg:
                    raise StepFailed(
                        "the notification does not say it was only composed — "
                        "claiming delivery would be false")
                if "none of them held" in msg and "settled" in msg:
                    raise StepFailed("the message contradicts itself")
                page.locator(".escsent").scroll_into_view_if_needed()
                page.wait_for_timeout(400)
                shot(page, "escalation-notify",
                     "the message, composed for the rider to send")
                findings.append(
                    "escalation fired after 4 attempts and composed a manager "
                    "notification without sending anything")
            else:
                findings.append("NOTE: escalation offered no NOTIFY MANAGER button")
        else:
            findings.append(
                "NOTE: the retry budget was spent but no escalation appeared")

        # -- 4. the reveal, and only now -----------------------------------
        reveal_btn = page.locator("button.reveal-cta").first
        if not reveal_btn.count():
            raise StepFailed("no reveal button appeared after the booking")
        reveal_btn.click()
        page.wait_for_selector(".revealbox", timeout=20_000)
        page.wait_for_timeout(900)
        shot(page, "reveal-top", "what actually happened")

        # The comparison column must never be crowned green unless it is
        # genuinely cheaper in expectation. Showing a dearer option as "what
        # the engine picked" tells the viewer the opposite of the point.
        head = page.locator(".cmp2 thead th")
        if head.count() >= 3:
            alt_label = head.nth(2).inner_text()
            costs = page.locator(".cmp2 tr.big td")
            chosen_cost = money(costs.nth(1).inner_text())
            alt_cost = money(costs.nth(2).inner_text())
            crowned = "win" in (head.nth(2).get_attribute("class") or "")
            print(f"       reveal: chose ₹{chosen_cost:.0f} vs "
                  f"{alt_label.splitlines()[0]} ₹{alt_cost:.0f} "
                  f"({'crowned' if crowned else 'not crowned'})")
            if crowned and alt_cost >= chosen_cost:
                raise StepFailed(
                    f"the reveal crowns a MORE expensive option: "
                    f"₹{alt_cost:.0f} shown as the winner against ₹{chosen_cost:.0f}")
            if crowned:
                findings.append(
                    f"crossover shown: chosen ₹{chosen_cost:.0f} expected vs "
                    f"alternative ₹{alt_cost:.0f} — the product's whole point")
        page.locator(".cmp2").scroll_into_view_if_needed()
        page.wait_for_timeout(400)
        shot(page, "reveal-comparison",
             "advertised vs expected, side by side")

        why = page.locator(".reveal-why li")
        if why.count() == 0:
            raise StepFailed("the reveal produced no explanation")
        print(f"       reveal explanation: {why.count()} sentences")
        why.first.scroll_into_view_if_needed()
        page.wait_for_timeout(400)
        shot(page, "reveal-explanation", "the engine explains itself")

        # -- 5. insights ---------------------------------------------------
        page.locator("button.reveal-cta", has_text="View mobility insights").click()
        page.wait_for_selector(".panel-card", timeout=25_000)
        page.wait_for_timeout(900)
        shot(page, "insights", "the market behind the failure")
        page.locator(".panel-card").nth(2).scroll_into_view_if_needed()
        page.wait_for_timeout(500)
        shot(page, "insights-relationships",
             "trip length vs acceptance, labelled association")

        # -- 6. the engine -------------------------------------------------
        page.locator("nav.viewnav button", has_text="Intelligence").click()
        page.wait_for_selector(".cmp-form", timeout=20_000)
        page.locator("button.go", has_text="Compare").click()
        page.wait_for_selector(".ocard", timeout=40_000)
        page.wait_for_timeout(900)
        titles = page.locator(".ocard-head h3")
        seen: dict[str, int] = {}
        for i in range(titles.count()):
            t = titles.nth(i).inner_text().strip()
            seen[t] = seen.get(t, 0) + 1
        for name, count in seen.items():
            if count > 1:
                vias = page.locator(".ocard-via").count()
                if vias < count:
                    raise StepFailed(
                        f"{count} cards all titled {name!r} with nothing to "
                        f"tell them apart")
        shot(page, "intelligence", "expected cost across every option")

        # -- 7. enterprise -------------------------------------------------
        page.locator("nav.viewnav button", has_text="Enterprise").click()
        page.wait_for_selector(".kpi", timeout=40_000)
        page.wait_for_timeout(1200)
        shot(page, "enterprise", "the same problem at organisation scale")
        if page.locator(".enttable").count():
            page.locator(".enttable").first.scroll_into_view_if_needed()
            page.wait_for_timeout(500)
            shot(page, "enterprise-scorecard",
                 "providers ranked by what a km actually costs")
        # Database keys are for joins. `bike_taxi` in a sentence an operations
        # lead reads is a leak, and it read exactly like one.
        #
        # The audit log is exempt on purpose: it is a machine record of what the
        # system decided, and a stable identifier is the right thing to write
        # there. Dressing it up as a display label would make the trail worse.
        cards = page.locator(".card:not(.auditcard)")
        ent_text = NL.join(cards.nth(i).inner_text()
                           for i in range(cards.count()))
        for key in ("bike_taxi", "namma_yatri", "provider_id"):
            if key in ent_text:
                raise StepFailed(f"raw identifier {key!r} is on the enterprise page")

        last = page.locator(".card").last
        last.scroll_into_view_if_needed()
        page.wait_for_timeout(500)
        shot(page, "governance", "every AI decision, recorded")

        # -- 8. the planner still works ------------------------------------
        page.locator("nav.viewnav button", has_text="Journey planner").click()
        page.wait_for_selector(".panel", timeout=20_000)
        page.wait_for_timeout(700)
        shot(page, "journey-planner", "the original multi-modal planner")

        if console_errors:
            raise StepFailed("the page logged errors:" + NL + "    "
                             + (NL + "    ").join(console_errors[:6]))
        if failed_requests:
            raise StepFailed("requests failed:" + NL + "    "
                             + (NL + "    ").join(failed_requests[:6]))
        print("       no console errors, no failed requests")

        browser.close()

    index = OUT / "README.md"
    index.write_text(
        "# Demo screenshots\n\n"
        "Captured by `python scripts/capture_demo.py` against a live server. "
        "Each step was asserted before it was photographed.\n\n"
        + "\n".join(f"{i+1}. **{name}** — {cap}" for i, (name, cap) in enumerate(shots))
        + "\n",
        encoding="utf-8")

    print(f"\n  {len(shots)} screenshots -> {OUT}")
    if findings:
        print("\n  FINDINGS")
        for f in findings:
            print(f"    - {f}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except StepFailed as exc:
        print(f"\n  STEP FAILED: {exc}", file=sys.stderr)
        sys.exit(1)
