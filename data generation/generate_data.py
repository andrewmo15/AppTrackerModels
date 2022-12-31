import random
import csv

companyfile = open(file="datasets/companies.txt", mode="r")
namesfile = open(file="datasets/names.txt", mode="r")
positionsfile = open(file="datasets/positions.txt", mode="r")
rejectionsfile = open(file="prompts/rejection_prompts.txt", mode="r")
submissionsfile = open(file="prompts/submission_prompts.txt", mode="r")

companies = companyfile.read().split("\n")
names = namesfile.read().split("\n")
positions = positionsfile.read().split("\n")

rejection_prompts = rejectionsfile.read().split("--------------\n")
rejection_subjects = rejection_prompts[0].split("\n")[1:-1]
rejection_intros = rejection_prompts[1].split("\n")[1:-1]
rejection_bodies = rejection_prompts[2].split("\n")[1:-1]
rejection_closing = rejection_prompts[3].split("\n")[1:]
rejection_final = rejection_prompts[4].split("\n")[1:-1]

submission_prompts = submissionsfile.read().split("--------------\n")
submission_subjects = submission_prompts[0].split("\n")[1:-1]
submission_optional = submission_prompts[1].split("\n")[1:]
submission_optional_start = submission_optional + ["" for _ in range(len(submission_optional) // 2)]
submission_intros = submission_prompts[2].split("\n")[1:-1]
submission_stuffer = submission_prompts[3].split("\n")[1:-1]
submission_bodies = submission_prompts[4].split("\n")[1:-1]
submission_closing = submission_prompts[5].split("\n")[1:]
submission_final = submission_prompts[6].split("\n")[1:-1]

senderemails = ["no-reply@us.greenhouse-mail.io", "+autoreply@talent.icims.com", "no-reply@", "companyemail", "no-reply@hire.lever.co", "@myworkday.com"]

def generate_emails(companies, names, positions, subjects, prompts, emailtype, data_size):
    rows = []
    prompts_copy = [prompt.copy() for prompt in prompts]
    companies_copy = companies.copy()
    names_copy = names.copy()
    positions_copy = positions.copy()
    subjects_copy = subjects.copy()
    for _ in range(data_size):
        if len(names_copy) <= 3:
            names_copy = names.copy()
        if len(companies_copy) <= 0:
            companies_copy = companies.copy()
        if len(positions_copy) <= 0:
            positions_copy = positions.copy()
        if len(subjects_copy) <= 0:
            subjects_copy = subjects.copy()
        for i, prompt in enumerate(prompts_copy):
            if len(prompt) <= 0:
                prompts_copy[i] = prompts[i].copy()

        promptsidx = [random.randint(0, len(prompt) - 1) for prompt in prompts_copy]
        nameidx = random.randint(0, len(names_copy) - 1)
        jobidx = random.randint(0, len(positions_copy) - 1)
        subjectidx = random.randint(0, len(subjects_copy) - 1)
        companyidx = random.randint(0, len(companies_copy) - 1)

        candidatefirstname = names_copy[nameidx]
        candidatelastname = names_copy[(nameidx + 1) % (len(names_copy) - 1)]
        recruiterfirstname = names_copy[(nameidx + 2) % (len(names_copy) - 1)]
        recruiterlastname = names_copy[(nameidx + 3) % (len(names_copy) - 1)]
        candidatename = candidatefirstname + " " + candidatelastname
        recruitername = recruiterfirstname + " " + recruiterlastname
        company = companies_copy[companyidx]
        job = positions_copy[jobidx]
        email = []
        for i, idx in enumerate(promptsidx):
            email.append(prompts_copy[i][idx])
        email = " ".join(email)
        email = email.replace("[*Candidate]", candidatename)
        email = email.replace("[*Position]", job)
        email = email.replace("[*Recruiter]", recruitername)
        email = email.replace("[*Company]", company)
        email.strip()

        subject = subjects_copy[subjectidx]
        subject = subject.replace("[*Candidate]", candidatename)
        subject = subject.replace("[*Position]", job)
        subject = subject.replace("[*Recruiter]", recruitername)
        subject = subject.replace("[*Company]", company)
        
        companyfix = company.lower()
        companyfix = "".join(companyfix.split())
        sender = senderemails[random.randint(0, len(senderemails) - 1)]
        name = recruitername.lower()
        name = "".join(name.split())
        if sender == "+autoreply@talent.icims.com":
            sender = company + " <" + companyfix + sender + ">"
        elif sender == "no-reply@":
            sender = company + " <" + sender + companyfix + ".com>"
        elif sender == "companyemail": 
            sender = recruitername + " <" + name + "@" + companyfix + ".com>"
        elif sender == "@myworkday.com":
            sender = company + " <" + companyfix + sender + ">"

        del names_copy[nameidx]
        del positions_copy[jobidx]
        del companies_copy[companyidx]
        del subjects_copy[subjectidx]
        for i, idx in enumerate(promptsidx):
            del prompts_copy[i][idx]
        row = [subject, sender, email, emailtype, company, job]
        rows.append(row)
    return rows

with open("data.csv", 'w', encoding='UTF8') as f:
    header = ['subject', 'from', 'body', "status", "company", "position"]
    writer = csv.writer(f)
    writer.writerow(header)
    reject_prompts = [rejection_intros, rejection_bodies, rejection_closing, rejection_final]
    rows = generate_emails(companies, names, positions, rejection_subjects, reject_prompts, "REJECTED", 20000)
    for row in rows:
        writer.writerow(row)
    submit_prompts = [submission_optional_start, submission_intros, submission_stuffer, submission_bodies, submission_closing, submission_final]
    rows = generate_emails(companies, names, positions, submission_subjects, submit_prompts, "SUBMITTED", 20000)
    for row in rows:
        writer.writerow(row)
    
companyfile.close()
namesfile.close()
positionsfile.close()
rejectionsfile.close()
submissionsfile.close()