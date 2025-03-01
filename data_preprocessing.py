# This is almost the same file as HW2, just modified to filter out {Joy, Sadness}

import csv
import re

from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build

# I already have a google service account from past projects
# make sure to create one following the instructions here
#
# https://cloud.google.com/iam/docs/service-accounts-create#console
# https://console.cloud.google.com/projectselector/iam-admin/serviceaccounts/create?walkthrough_id=iam--create-service-account#step_index=1
#
# then download the corresponding credentials.json for it
#
# ONLY NECESSARY IF YOU WANT TO RUN THIS FILE PROPERLY, OTHERWISE PROCESSED DATA IS ALREADY IN .csv PROVIDED

SERVICE_ACCOUNT_FILE = "credentials.json"

# Unchanged from HW2
# https://docs.google.com/spreadsheets/d/19Dg3yKvUR3XCyiAsfV9UeSU_7SRgylXwS6GQSwU0AN4/edit?gid=0#gid=0
SPREADSHEET_ID = "19Dg3yKvUR3XCyiAsfV9UeSU_7SRgylXwS6GQSwU0AN4"

SHEET_NAMES = ["Sheet1"]
OUTPUT_PATH = "cs173-hw3-processed.csv"

SCOPES = ["https://www.googleapis.com/auth/spreadsheets.readonly"]
credentials = Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE, scopes=SCOPES)
service = build("sheets", "v4", credentials=credentials)


def clean_text(string: str) -> str:
    string = re.sub(r"\s+", " ", string)
    return string.strip()


def extract_emotions(headers: list[str]) -> dict[int, list[str]]:
    emotions = {}

    for i, header in enumerate(headers):
        header = re.sub(r"\s*Sentence?s\s*", "", header)
        emotions[i] = [emotion.strip() for emotion in header.split("+")]

    return emotions


def process_spreadsheet():
    sheet = (
        service.spreadsheets()
        .values()
        .get(spreadsheetId=SPREADSHEET_ID, range=SHEET_NAMES[0])
        .execute()
    )
    rows: list[str] = sheet.get("values", [])

    headers = rows[0]
    emotions = extract_emotions(headers)

    processed_rows = []
    for row_i, row in enumerate(rows[1:], start=1):
        for col_j, cell in enumerate(row):
            if col_j % 2 == 1:
                trim_text = clean_text(cell)
                if len(trim_text) > 0:
                    for emotion in emotions[col_j]:
                        if emotion in ["Joy", "Sadness"]:
                            processed_rows.append([row_i, emotion, trim_text])

    with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as f:
        # Output Format
        # Row #, Emotion (Class), Paragraph

        writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
        writer.writerows(processed_rows)


if __name__ == "__main__":
    process_spreadsheet()
