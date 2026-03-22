import argparse
import json
import urllib.error
import urllib.request


def main() -> None:
    parser = argparse.ArgumentParser(description="Test /predict endpoint")
    parser.add_argument("--url", default="http://127.0.0.1:5000/predict")
    parser.add_argument("--attendance_percentage", type=float, default=85)
    parser.add_argument("--tasks_completed", type=float, default=12)
    parser.add_argument("--tasks_pending", type=float, default=3)
    parser.add_argument("--avg_task_score", type=float, default=8.2)
    parser.add_argument("--mentor_feedback_score", type=float, default=8.5)
    parser.add_argument("--communication_score", type=float, default=8.0)
    parser.add_argument("--teamwork_score", type=float, default=8.1)
    parser.add_argument("--punctuality_score", type=float, default=8.4)
    parser.add_argument("--learning_progress", type=float, default=8.3)
    args = parser.parse_args()

    payload = {
        "attendance_percentage": args.attendance_percentage,
        "tasks_completed": args.tasks_completed,
        "tasks_pending": args.tasks_pending,
        "avg_task_score": args.avg_task_score,
        "mentor_feedback_score": args.mentor_feedback_score,
        "communication_score": args.communication_score,
        "teamwork_score": args.teamwork_score,
        "punctuality_score": args.punctuality_score,
        "learning_progress": args.learning_progress,
    }

    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        args.url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(request) as response:
            body = response.read().decode("utf-8")
            print(f"STATUS={response.status}")
            print(body)
    except urllib.error.HTTPError as error:
        body = error.read().decode("utf-8")
        print(f"STATUS={error.code}")
        print(body)


if __name__ == "__main__":
    main()
