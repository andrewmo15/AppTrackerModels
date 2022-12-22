import imaplib
import email
import csv
 
# user = 'aliksemelianov@gmail.com'
# password = 'xqjybjodcaqktkcl'
user = 'andrewmoasdf@gmail.com'
password = 'rwkixnairelyitli'
imap_url = 'imap.gmail.com'

mail = imaplib.IMAP4_SSL(imap_url)
mail.login(user, password)
mail.select('Inbox')
_, selected_mails = mail.search(None, 'ALL')


emails = []
for num in selected_mails[0].split():
    _, data = mail.fetch(num , '(RFC822)')
    _, bytes_data = data[0]

    email_message = email.message_from_bytes(bytes_data)
    data = {}
    data["subject"] = email_message["subject"]
    data["to"] = email_message["to"]
    data["from"] = email_message["from"]
    data["date"] = email_message["date"]

    data["body"] = ""
    for part in email_message.walk():
        if part.get_content_type()=="text/plain" or part.get_content_type()=="text/html":
            message = part.get_payload(decode=True)
            data["body"] += message.decode()
            break
    data["status"] = ""
    data["company"] = ""
    emails.append(data)

with open('emaildata.csv', 'w', encoding='UTF8') as f:
    header = ['subject', 'to', 'from', 'date', 'body', "status", "company"]
    writer = csv.writer(f)
    writer.writerow(header)
    for emaildata in emails:
        temp = []
        for heading in header:
            temp.append(emaildata[heading])
        writer.writerow(temp)