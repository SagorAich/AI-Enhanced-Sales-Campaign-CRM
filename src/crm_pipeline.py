import os
import argparse
import random
import time
from typing import Dict
import smtplib
from email.message import EmailMessage

import pandas as pd
from tqdm import tqdm

from src.llm_client import LLMClient


def enrich_lead(llm: LLMClient, lead: Dict) -> Dict:
    prompt = (
        "You are a helpful assistant that enriches a sales lead.\n"
        f"Known fields: first_name={lead.get('first_name')}, last_name={lead.get('last_name')}, email={lead.get('email')}, company={lead.get('company')}, title={lead.get('title')}, industry={lead.get('industry')}, location={lead.get('location')}.\n"
        "Fill any missing fields (company, title, industry, location) with short plausible values if blank, and then suggest a concise buyer persona label (one or two words), and a 1-2 sentence persona description.\n"
        "Return output as lines in the format: company:..., title:..., industry:..., location:..., persona:..., persona_desc:...\n"
    )
    out = llm.generate(prompt)
    parsed = {}
    for line in out.splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            parsed[k.strip()] = v.strip()
    # fallbacks
    lead['company'] = lead.get('company') or parsed.get('company') or lead.get('company') or ''
    lead['title'] = lead.get('title') or parsed.get('title') or ''
    lead['industry'] = lead.get('industry') or parsed.get('industry') or ''
    lead['location'] = lead.get('location') or parsed.get('location') or ''
    lead['persona'] = parsed.get('persona', 'Unknown')
    lead['persona_desc'] = parsed.get('persona_desc', '')
    return lead


def score_lead(llm: LLMClient, lead: Dict) -> Dict:
    prompt = (
        "You are an expert sales analyst. Given this lead, provide a numeric priority 1-5 (5 highest) and one short reason.\n"
        f"Lead: first_name={lead.get('first_name')}, last_name={lead.get('last_name')}, company={lead.get('company')}, title={lead.get('title')}, industry={lead.get('industry')}, persona={lead.get('persona')}.\n"
        "Return: priority: <1-5>\nreason: <one sentence>"
    )
    out = llm.generate(prompt)
    priority = 3
    reason = ''
    for line in out.splitlines():
        if line.lower().strip().startswith('priority'):
            try:
                priority = int(''.join(filter(str.isdigit, line)))
            except Exception:
                priority = 3
        if line.lower().strip().startswith('reason'):
            _, r = line.split(':', 1)
            reason = r.strip()
    lead['priority'] = priority
    lead['priority_reason'] = reason
    return lead


def draft_email(llm: LLMClient, lead: Dict) -> Dict:
    prompt = (
        "Write a short, personalized outreach email (subject + body) in a friendly professional tone. Keep it <100 words in body. \n"
        f"Lead: {lead.get('first_name')} {lead.get('last_name')}, title={lead.get('title')}, company={lead.get('company')}, persona={lead.get('persona')}.\n"
        "Include a single short sentence clear call-to-action.\n"
        "Return exactly as:\nSubject: <subject line>\n\nBody: <email body>"
    )
    out = llm.generate(prompt)
    subject = ''
    body = ''
    if 'Subject:' in out:
        parts = out.split('Subject:', 1)[1]
        if 'Body:' in parts:
            subj, bod = parts.split('Body:', 1)
            subject = subj.strip()
            body = bod.strip()
        else:
            subject = parts.strip()
    else:
        # fallback: whole text is body
        body = out.strip()
        subject = f"Quick note for {lead.get('first_name')}"
    lead['email_subject'] = subject
    lead['email_body'] = body
    return lead


def send_email(smtp_host: str, smtp_port: int, from_addr: str, to_addr: str, subject: str, body: str):
    msg = EmailMessage()
    msg['From'] = from_addr
    msg['To'] = to_addr
    msg['Subject'] = subject
    msg.set_content(body)
    with smtplib.SMTP(smtp_host, smtp_port) as s:
        s.send_message(msg)


def simulate_and_classify_responses(llm: LLMClient, lead: Dict) -> Dict:
    # Simulate a reply for demo purposes for some leads
    # We'll simulate replies for higher-priority leads with some randomness
    simulate = False
    if lead.get('priority', 3) >= 4 and random.random() < 0.8:
        simulate = True
    elif lead.get('priority', 3) == 3 and random.random() < 0.25:
        simulate = True

    if not simulate:
        lead['status'] = 'no_response'
        lead['response_text'] = ''
        lead['response_category'] = 'No Response'
        return lead

    prompt_reply = (
        "You are the prospect receiving this email. Write a short reply (1-3 sentences) that reflects a realistic reaction: interested, maybe later, or not interested.\n"
        f"Email subject: {lead.get('email_subject')}\nEmail body: {lead.get('email_body')}\nPersona: {lead.get('persona')}\n"
    )
    reply = llm.generate(prompt_reply)
    # classify the reply
    prompt_class = (
        "Classify the following reply into one of: Interested, Maybe, Not Interested. Return only the one-word label.\n"
        f"Reply: {reply}\n"
    )
    label = llm.generate(prompt_class)
    label_clean = label.splitlines()[0].strip().split()[0]

    lead['status'] = 'replied'
    lead['response_text'] = reply
    lead['response_category'] = label_clean
    return lead


def generate_report(llm: LLMClient, df: pd.DataFrame, out_path: str):
    total = len(df)
    sent = df['status'].isin(['sent','replied']).sum()
    replied = (df['status'] == 'replied').sum()
    # Ensure priority is treated as numeric (some rows may contain empty strings or error text)
    # Coerce non-numeric values to NaN and compute mean safely.
    if 'priority' in df.columns:
        priorities = pd.to_numeric(df['priority'], errors='coerce')
        avg_priority = float(priorities.mean(skipna=True)) if not priorities.empty else 0.0
    else:
        avg_priority = 0.0

    persona_counts = df.get('persona', pd.Series([], dtype=object)).fillna('Unknown').value_counts().to_dict()

    summary = (
        f"Campaign summary:\n\nTotal leads: {total}\nSent: {sent}\nReplied: {replied}\nAverage priority: {avg_priority:.2f}\n\nPersona breakdown:\n"
    )
    for p, c in persona_counts.items():
        summary += f"- {p}: {c}\n"

    # Ask LLM for insights
    prompt_insights = (
        "You are a smart sales analyst. Given the campaign summary below, write a short markdown report (3-6 paragraphs) with insights, suggestions to improve outreach, and 3 quick action items.\n\n"
        f"{summary}"
    )
    insights = llm.generate(prompt_insights, max_new_tokens=256)

    with open(out_path, 'w', encoding='utf-8') as f:
        f.write('# Campaign Summary\n\n')
        f.write(summary + '\n\n')
        f.write('## AI Insights\n\n')
        f.write(insights)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', default='data/leads.csv')
    parser.add_argument('--out', default='data/leads_out.csv')
    parser.add_argument('--smtp-host', default='localhost')
    parser.add_argument('--smtp-port', type=int, default=1025)
    parser.add_argument('--from-addr', default='noreply@example.com')
    parser.add_argument('--model', default='openai/gpt-oss-20b')
    parser.add_argument('--api-key', default=None, help='Groq API key (also reads GROQ_API_KEY env var)')
    args = parser.parse_args()

    llm = LLMClient(api_key=args.api_key, model=args.model)

    df = pd.read_csv(args.csv)

    # Ensure output columns exist
    for col in ['persona', 'persona_desc', 'priority', 'priority_reason', 'email_subject', 'email_body', 'status', 'response_text', 'response_category']:
        if col not in df.columns:
            df[col] = ''

    rows = df.to_dict(orient='records')
    processed = []
    for lead in tqdm(rows, desc='Processing leads'):
        try:
            lead = enrich_lead(llm, lead)
            lead = score_lead(llm, lead)
            lead = draft_email(llm, lead)
            # send email
            try:
                send_email(args.smtp_host, args.smtp_port, args.from_addr, lead.get('email'), lead['email_subject'], lead['email_body'])
                lead['status'] = 'sent'
            except Exception as e:
                lead['status'] = f'send_error: {e}'
            # simulate and classify responses for demo
            lead = simulate_and_classify_responses(llm, lead)
        except Exception as e:
            lead['status'] = f'error: {e}'
        processed.append(lead)
        # brief pause to avoid HF rate limits
        time.sleep(0.5)

    out_df = pd.DataFrame(processed)
    out_df.to_csv(args.out, index=False)

    os.makedirs('reports', exist_ok=True)
    generate_report(llm, out_df, 'reports/campaign_summary.md')

    print('\nDone. Updated CSV at', args.out)
    print('Report at reports/campaign_summary.md')
    print('If using MailHog, open http://localhost:8025 to see sent messages')


if __name__ == '__main__':
    main()
