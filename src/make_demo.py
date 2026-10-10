from pathlib import Path
import pandas as pd
from datetime import datetime, timedelta, timezone
root=Path(__file__).resolve().parents[1]
# Each tuple is manually authored: text, sentiment, urgency, ambiguous.
# Fictional examples only; labels are never used by the demo extractor.
data={
'billing':[
("I've been charged twice this month.",'negative','high',0),
('Please send a copy of my September invoice.','neutral','medium',0),
('I am frustrated that my refund has not arrived.','negative','medium',0),
('Which payment methods do you accept?','neutral','medium',0),
('The subscription renewal price changed; can you explain it?','neutral','medium',0),
('My card was declined and my account is blocked today.','neutral','high',0),
('The VAT number on our invoice is incorrect.','neutral','medium',0),
('I need the receipt for our energy-monitoring subscription.','neutral','medium',0),
('The double charge is unacceptable. Please correct it immediately.','negative','high',0),
('Can our annual subscription be paid monthly instead?','neutral','medium',0),
('The discount promised by sales is missing from the bill.','negative','medium',0),
('Thanks for explaining the payment terms. Could you fix the incorrect invoice?','positive','medium',1),
('The app crashes, but my main concern is a duplicate payment.','negative','high',1),
('I am not sure whether this charge is yours.','neutral','medium',1),
('Could you explain this fee?','neutral','medium',0),
('Our payment deadline is today; please correct the invoice before then.','neutral','high',0)],
'technical':[
('The app crashes every time I open the meter dashboard. I am frustrated.','negative','medium',0),
('I cannot login and need access urgently for a meeting.','neutral','high',0),
('The energy meter is offline. Please help restore the connection.','neutral','medium',0),
('The CSV download fails with error 503.','neutral','medium',0),
('The dashboard is not loading on my tablet.','neutral','medium',0),
('The hourly usage chart is stuck on yesterday.','neutral','medium',0),
('Password reset emails are not arriving.','neutral','medium',0),
('The sensor readings are not syncing with our account.','neutral','medium',0),
('A security bug exposes another building on my dashboard; fix it immediately.','neutral','high',0),
('Our operations team is blocked because the app is broken.','neutral','high',0),
('The alert settings do not work after the update.','neutral','medium',0),
('The dashboard fails to render. A dark mode would also be nice.','neutral','medium',1),
('Thanks for the great app, but the export button is broken.','positive','medium',1),
('Maybe the meter is offline, or perhaps I chose the wrong site.','neutral','medium',1),
('The browser shows a blank page after I sign in.','neutral','medium',0),
('The app keeps logging me out. This is annoying.','negative','medium',0)],
'feature_request':[
('Please add dark mode to the energy dashboard.','neutral','low',0),
('Could you add an integration with our building management system?','neutral','low',0),
('I would like weekly energy reports by email.','neutral','low',0),
('Please add an option to compare two sites.','neutral','low',0),
('I wish the chart could show carbon intensity alongside usage.','neutral','low',0),
('Feature request: let us export data in parquet format.','neutral','low',0),
('Could you add custom thresholds for high energy consumption?','neutral','low',0),
('Please add support for time-of-use tariff comparisons.','neutral','low',0),
('I suggest a dashboard widget for solar generation.','neutral','low',0),
('We would like role-based permissions for site managers.','neutral','low',0),
('Please add a way to annotate equipment maintenance on charts.','neutral','low',0),
('I love the app. Please add an integration with Home Assistant.','positive','low',1),
('Could you add an invoice preview feature? This is a suggestion, not a payment issue.','neutral','low',1),
('Maybe a simpler setup wizard would help; I am not reporting a failure.','neutral','low',1),
('I would like a forecast of next week’s building energy demand.','neutral','low',0),
('Could you add a feature to group meters by floor?','neutral','low',0)],
'praise':[
('Thank you for the clear energy reports.','positive','low',0),
('I love the new dashboard design.','positive','low',0),
('The weekly insights are excellent.','positive','low',0),
('Great work on the meter comparison view.','positive','low',0),
('Your support team was wonderful.','positive','low',0),
('The onboarding guide is brilliant.','positive','low',0),
('Fantastic product; it helped us identify wasted energy.','positive','low',0),
('I am happy with the clarity of the charts.','positive','low',0),
('Thanks for the quick response yesterday.','positive','low',0),
('The new navigation feels great.','positive','low',0),
('Excellent documentation and a very helpful team.','positive','low',0),
('Thanks for fixing the login issue; everything is working now.','positive','low',1),
('The new invoice screen is great; no help is needed.','positive','low',1),
('I was unsure before, but now I love the monthly report.','positive','low',1),
('Great job making energy consumption easier to understand.','positive','low',0),
('Thank you!','positive','low',0)],
'other':[
('What are your office hours?','neutral','low',0),
('Where is your company based?','neutral','low',0),
('Do you have a press contact?','neutral','low',0),
('Can I speak to someone about a partnership?','neutral','low',0),
('I sent this message to the wrong company.','neutral','low',0),
('What languages does your documentation come in?','neutral','low',0),
('How can I apply for a job at your company?','neutral','low',0),
('Do you exhibit at the energy conference in Berlin?','neutral','low',0),
('Can you point me to your accessibility statement?','neutral','low',0),
('Where can I find your company registration number?','neutral','low',0),
('This is a test message from our training workshop.','neutral','low',0),
('Maybe I meant to contact another service. I am not sure.','neutral','low',1),
('Hello.','neutral','low',1),
('Something about energy, perhaps.','neutral','low',1),
('Could you explain what this service does?','neutral','low',0),
('I found your business through a conference directory.','neutral','low',0),
('???','neutral','low',1)]}
rows=[];labels=[];n=0
for category,examples in data.items():
 for text,sentiment,urgency,ambiguous in examples:
  n+=1;tid=f'DEMO-{n:03d}'
  date=datetime(2025,1,1,9,tzinfo=timezone.utc)+timedelta(days=round((n-1)*364/80))
  rows.append(dict(ticket_id=tid,submitted_at=date.isoformat(),raw_text=text))
  labels.append(dict(ticket_id=tid,category=category,sentiment=sentiment,urgency=urgency,is_ambiguous=bool(ambiguous),label_source='author_synthetic_demo'))
pd.DataFrame(rows).to_csv(root/'data/demo/support_tickets.csv',index=False)
pd.DataFrame(labels).to_csv(root/'data/demo/support_tickets_labeled.csv',index=False)
assert n==81
