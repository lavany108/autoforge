"""
Phase 5 Verification Script
Navigates through all 6 pages of the Streamlit dashboard using Playwright,
verifies rendering of each component, and captures high-resolution screenshots
into docs/screenshots/.
"""

import os
import sys
import time
from playwright.sync_api import sync_playwright

SCREENSHOTS_DIR = "docs/screenshots"
os.makedirs(SCREENSHOTS_DIR, exist_ok=True)


def verify_phase5():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(viewport={"width": 1440, "height": 960})
        page = context.new_page()

        print("Connecting to Streamlit dashboard at http://localhost:8501...")
        page.goto("http://localhost:8501", timeout=30000)
        page.wait_for_load_state("networkidle")
        time.sleep(3)  # Allow Streamlit initial script to execute

        pages_to_test = [
            ("1. 📂 Upload & Profiling", "01_upload_profiling.png", "Dataset Ingestion & Automated Profiling"),
            ("2. 🚀 Automated Training", "02_training.png", "AutoML Training Engine"),
            ("3. 🏆 Experiment Leaderboard", "03_leaderboard.png", "Experiment Leaderboard"),
            ("4. 🔍 Model Explainability", "04_explainability.png", "Model Explainability & Feature Attribution"),
            ("5. 🚢 Model Deployment", "05_deployment.png", "Model Registry & Live Deployment"),
            ("6. 📊 Monitoring & Drift", "06_monitoring.png", "Data Drift Monitoring & Auto-Retraining"),
        ]

        for nav_label, img_name, header_text in pages_to_test:
            print(f"\nNavigating to: '{nav_label}'...")

            # Click navigation radio in sidebar
            # Streamlit radio labels can be selected by text
            radio_option = page.locator(f"label:has-text('{nav_label}')")
            if radio_option.count() > 0:
                radio_option.first.click()
            else:
                # Fallback click by span or text
                page.click(f"text={nav_label}")

            time.sleep(3)  # Allow Streamlit rerun
            page.wait_for_load_state("networkidle")

            # Verify page header rendered
            assert page.locator(f"text={header_text}").count() > 0, f"Header '{header_text}' not found on page"
            print(f"  Verified header: '{header_text}'")

            # Capture screenshot
            out_img = os.path.join(SCREENSHOTS_DIR, img_name)
            page.screenshot(path=out_img, full_page=False)
            assert os.path.exists(out_img), f"Screenshot not saved: {out_img}"
            file_kb = os.path.getsize(out_img) / 1024
            print(f"  Captured screenshot: {out_img} ({file_kb:.1f} KB)")

        browser.close()
        print("\n>>> PHASE 5 VERIFICATION PASSED: All 6 pages rendered, verified, and captured to docs/screenshots/ <<<")


if __name__ == "__main__":
    verify_phase5()
