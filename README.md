# AI-powered Sales Campaign CRM (Local MVP)

Lightweight Python MVP to ingest leads from CSV, use the Groq API (free SDK) to enrich/score/draft outreach, send emails via SMTP (MailHog recommended for local demo), and write results + AI-written campaign report.

Overview
- Input: `data/leads.csv` (sample provided with 25 leads)
- Output: updated CSV `data/leads_out.csv` (adds persona, priority, subject, body, status, response_text, response_category)
- Emails: sent to `smtp_host`/`smtp_port` (defaults to MailHog: localhost:1025)
- Report: `reports/campaign_summary.md` (AI-written summary & insights)

Quick start (Windows / PowerShell)
1. Install Python packages:

```powershell
python -m pip install -r requirements.txt
```

2. Start MailHog (for local demo)
- Download MailHog for Windows: https://github.com/mailhog/MailHog/releases
- Run MailHog binary. It listens by default on SMTP `localhost:1025` and web UI `http://localhost:8025`

3. Run the pipeline (default uses sample CSV)

```powershell
python -m src.crm_pipeline
```

4. Inspect results
- MailHog web UI: http://localhost:8025 to view sent emails
- Updated CSV: `data/leads_out.csv`
- Report: `reports/campaign_summary.md`

Configuration
- To provide your Groq API key for this session, either set the `GROQ_API_KEY` environment variable or pass the key via the CLI using `--api-key`.
 - To use a different SMTP server, use `--smtp-host` and `--smtp-port`.

Notes
- Default Groq model is `openai/gpt-oss-20b`. You can override it with `--model` if needed.
- This MVP simulates a subset of responses locally and classifies them with the same LLM (so you get response classifications in the CSV for demo). If you have a real reply inbox, we can add IMAP polling to classify real replies.

Files of interest
- `src/llm_client.py` - Groq-based LLM client
- `src/crm_pipeline.py` - the pipeline orchestrator
- `data/leads.csv` - sample leads
- `reports/campaign_summary.md` - generated after run

Campaign summary:

Total leads: 20 Sent: 9 Replied: 9 Average priority: 3.60

Persona breakdown:

Unknown: 20
AI Insights
Campaign Snapshot

Leads: 20
Sent: 9 (45 %)
Replied: 9 (100 % of those contacted)

