import imaplib
import email
import csv
# import html2text
# SWTICHED TO BEAUTIFUL SOUP
from bs4 import BeautifulSoup

 
# user = 'aliksemelianov@gmail.com'
# password = 'xqjybjodcaqktkcl'
user = 'andrewmoasdf@gmail.com'
password = 'rwkixnairelyitli'
imap_url = 'imap.gmail.com'
mail = imaplib.IMAP4_SSL(imap_url)
mail.login(user, password)
mail.select('Inbox')
_, selected_mails = mail.search(None, 'ALL')

def getTextFromHTML(html):
    soup = BeautifulSoup(html, features="html.parser")
    # rip out all scripts and style elements
    for script in soup(["script", "style"]):
        script.extract()    
    text = soup.get_text()
    # break into lines and remove leading and trailing space on each
    lines = (line.strip() for line in text.splitlines())
    # break multi-headlines into a line each
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    # drop blank lines
    text = '\n'.join(chunk for chunk in chunks if chunk)
    return text

emails = []
counter = 0
for num in selected_mails[0].split():
    _, data = mail.fetch(num , '(RFC822)')
    _, bytes_data = data[0]
    counter += 1
    if counter % 20 == 0: print(counter)
    email_message = email.message_from_bytes(bytes_data)
    data = {}
    data["subject"] = email_message["subject"]
    data["to"] = email_message["to"]
    data["from"] = email_message["from"]
    data["date"] = email_message["date"]
    if data["subject"] == "" or data["to"] == "" or data["from"] == "" or data["date"] == "" or user in data["from"]:
        continue

    data["body"] = ""
    for part in email_message.walk():
        if part.get_content_type()=="text/plain" or part.get_content_type()=="text/html":
            message = part.get_payload(decode=True)
            try:
                text = message.decode()
            except:
                text = message.decode("cp1252")
            if part.get_content_type()=="text/html":
                text = getTextFromHTML(text)
            data["body"] = " ".join(text.split())
            break
    if data["body"]=="":
        continue
    
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