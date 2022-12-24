import imaplib
import email
from email.header import decode_header, make_header
import csv
from bs4 import BeautifulSoup
import re

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

def cleanText(text, isHTML=False):
    # remove whitespace
    text = " ".join(text.split())
    # convert HTML to plain text
    if isHTML:
        text = getTextFromHTML(text)
    # remove links
    nolinks = re.sub(r'http\S+', '', text)
    # remove characters to prevent cell overflow
    return nolinks[:32760]

def hasEmptyParameters(email):
    try:
        return email["subject"] == "" or email["to"] == "" or email["from"] == "" or email["date"] == "" or email["subject"] == None or email["to"] == None or email["from"] == None or email["date"] == None
    except:
        return False

def getEmails(user, password, imap_url):
    mail = imaplib.IMAP4_SSL(imap_url)
    mail.login(user, password)
    mail.select('Inbox')
    _, selected_mails = mail.search(None, 'ALL')

    selected_mails = selected_mails[0].split()
    selected_mails.reverse()
    emails = []
    for num in selected_mails:
        _, data = mail.fetch(num , '(RFC822)')
        _, bytes_data = data[0]
        email_message = email.message_from_bytes(bytes_data)
        if hasEmptyParameters(email_message) or user in email_message["from"]:
            continue
        subject = str(make_header(decode_header(email_message["subject"])))
        to = str(make_header(decode_header(email_message["to"])))
        sender = str(make_header(decode_header(email_message["from"])))
        date = str(make_header(decode_header(email_message["date"])))
        data = {}
        data["subject"] = cleanText(subject)
        data["to"] = cleanText(to)
        data["from"] = cleanText(sender)
        data["date"] = cleanText(date)
        data["body"] = ""
        for part in email_message.walk():
            if part.get_content_type()=="text/plain" or part.get_content_type()=="text/html":
                message = part.get_payload(decode=True)
                # decode using correct charset
                try:
                    text = message.decode('utf-8', 'ignore')
                except:
                    text = message.decode("cp1252", 'ignore')
                data["body"] = cleanText(text, part.get_content_type()=="text/html")
                break
        if data["body"] == "" or data["body"] == None:
            continue
        data["status"] = ""
        data["company"] = ""
        emails.append(data)

    return emails

def createCSVDataset(filename, emails):
    with open(filename, 'w', encoding='UTF8') as f:
        header = ['subject', 'to', 'from', 'date', 'body', "status", "company"]
        writer = csv.writer(f)
        writer.writerow(header)
        for emaildata in emails:
            temp = []
            for heading in header:
                temp.append(emaildata[heading])
            writer.writerow(temp)

# user = 'aliksemelianov@gmail.com'
# password = 'xqjybjodcaqktkcl'
user = 'andrewmoasdf@gmail.com'
password = 'rwkixnairelyitli'
imap_url = 'imap.gmail.com'
emails = getEmails(user, password, imap_url)
createCSVDataset("emails.csv", emails)