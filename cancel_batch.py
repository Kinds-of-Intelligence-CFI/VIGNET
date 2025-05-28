import openai
from openai import OpenAI
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cancel a batch job.")
    parser.add_argument("--batch_id", type=str, help="The ID of the batch job to cancel.")
    args = parser.parse_args()
    client = OpenAI()

    if args.batch_id == "all":
        batches = client.batches.list()
        for batch in batches:
            if batch.status == "in_progress":
                print(f"Cancelling batch {batch.id}")
                client.batches.cancel(batch.id)
    else:
        client.batches.cancel(args.batch_id)
        print(f"Cancelled batch {args.batch_id}")
