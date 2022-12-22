import imaplib
import email
import csv
import html2text
 
# user = 'aliksemelianov@gmail.com'
# password = 'xqjybjodcaqktkcl'
user = 'andrewmoasdf@gmail.com'
password = 'rwkixnairelyitli'
imap_url = 'imap.gmail.com'
mail = imaplib.IMAP4_SSL(imap_url)
mail.login(user, password)
mail.select('Inbox')
_, selected_mails = mail.search(None, 'ALL')

h = html2text.HTML2Text()
h.ignore_links = True


emails = []
count = 0
for num in selected_mails[0].split():
    _, data = mail.fetch(num , '(RFC822)')
    _, bytes_data = data[0]
    count += 1
    if count % 20 == 0: 
        print(count)

    email_message = email.message_from_bytes(bytes_data)
    data = {}
    data["subject"] = email_message["subject"]
    data["to"] = email_message["to"]
    data["from"] = email_message["from"]
    if user in data["from"]:
        continue
    data["date"] = email_message["date"]

    data["body"] = ""
    
    for part in email_message.walk():
        if part.get_content_type()=="text/plain" or part.get_content_type()=="text/html":
            message = part.get_payload(decode=True)
            try:
                s = message.decode()
            except:
                s = message.decode("cp1252")
            if part.get_content_type()=="text/html":
                s = h.handle(s)
            data["body"] = " ".join(s.split())
            break
    data["status"] = ""
    data["company"] = ""
    emails.append(data)

with open('AlikEmailData.csv', 'w', encoding='UTF8') as f:
    header = ['subject', 'to', 'from', 'date', 'body', "status", "company"]
    writer = csv.writer(f)
    writer.writerow(header)
    for emaildata in emails:
        temp = []
        for heading in header:
            temp.append(emaildata[heading])
        writer.writerow(temp)