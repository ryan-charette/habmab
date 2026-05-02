"""Run the default portfolio demo."""

from habmab.cli import main


if __name__ == "__main__":
    raise SystemExit(
        main(
            [
                "--budget",
                "50",
                "--runs",
                "20",
                "--sites",
                "140",
                "--policies",
                "all",
                "--output",
                "reports/latest",
            ]
        )
    )
