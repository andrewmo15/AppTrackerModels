import imaplib
from email.header import decode_header, make_header
import csv
from bs4 import BeautifulSoup
import re
from fast_mail_parser import parse_email, ParseError

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
    text = " ".join(chunk for chunk in chunks if chunk)
    return text

def cleanText(text):
    text = str(make_header(decode_header(text)))
    # remove whitespace
    text = " ".join(text.split())
    # remove links
    text = re.sub(r'http\S+', '', text)
    # convert HTML to plain text
    text = getTextFromHTML(text)
    # remove characters to prevent cell overflow
    if len(text) > 32000:
        return text[:32000]
    return text

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
        # print(data)
        _, bytes_data = data[0]
        try:
            emaildata = parse_email(bytes_data)
        except ParseError as e:
            print("Failed to parse email: ", e)
        
        sender = ""
        for key, value in emaildata.headers.items():
            if key.lower() == "from":
                sender = value
                break

        data = {
            "subject": cleanText(emaildata.subject),
            "from": sender,
            "date": emaildata.date,
            "body": "",
            "status": "",
            "company": "",
        }
        if len(emaildata.text_plain) == 0 and len(emaildata.text_html) == 0:
            continue
        elif len(emaildata.text_plain) == 0:
            data["body"] = cleanText(emaildata.text_html[0])
            if not data["body"]:
                continue
        elif len(emaildata.text_html) == 0:
            data["body"] = cleanText(emaildata.text_plain[0])
            if not data["body"]:
                continue
        else:
            data["body"] = cleanText(emaildata.text_plain[0])
            if not data["body"]:
                data["body"] = cleanText(emaildata.text_html[0])
        if user in data["from"] or not (data["subject"] and data["from"] and data["date"] and data["body"]):
            continue
        emails.append(data)

    return emails

def createCSVDataset(filename, emails):
    with open(filename, 'w', encoding='UTF8') as f:
        header = ['subject', 'from', 'date', 'body', "status", "company"]
        writer = csv.writer(f)
        writer.writerow(header)
        for emaildata in emails:
            temp = []
            for heading in header:
                temp.append(emaildata[heading])
            writer.writerow(temp)

# user = 'aliksemelianov@gmail.com'
# password = 'xqjybjodcaqktkcl'
user = 'barnette@gmail.com'
password = 'cjaszgvtufajizyh'
# user = 'nathan.barnette@me.com'
# password = 'dpis-wdyt-ltpi-rqjn'
imap_url = 'imap.gmail.com'
# imap_url = 'imap.mail.me.com'
emails = getEmails(user, password, imap_url)
createCSVDataset("emails.csv", emails)