"""Builds eval_dataset.jsonl (105 questions) for the Asterion Employee Handbook.

Run from backend/:   python -m app.versions.gen.build_dataset
Then validate:       python -m app.versions.cli validate

Every answerable question carries `evidence` = verbatim snippets (with page numbers) that contain the answer.
Judge-free hit rate checks that at least one snippet is inside the chunks a version retrieved.
"""
import json
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "eval_dataset.jsonl"

Q = []


def add(cat, question, answer, evidence=()):
    Q.append({"category": cat, "question": question, "expected_answer": answer,
              "evidence": [{"page": p, "text": t} for p, t in evidence]})


S, M, H, U = "single_fact", "multi_part", "multi_hop", "unanswerable"

# ============================ SINGLE FACT (40) ============================
add(S, "What is the document ID of the employee handbook?", "ADS-HR-001.", [(1, "Document ID ADS-HR-001")])
add(S, "When does the handbook take effect?", "1 January 2027.", [(1, "Effective date 1 January 2027")])
add(S, "In which year was Asterion Digital Systems established?", "2017.", [(3, "established in 2017")])
add(S, "Where is the company's head office?", "Bengaluru, Karnataka, India.", [(1, "Head office Bengaluru, Karnataka, India")])
add(S, "Who is the Chief Technology Officer?", "Priya Verma.", [(5, "Chief Technology Officer (Priya Verma)")])
add(S, "What are the core office hours?", "9:30 AM to 6:30 PM.", [(7, "Core office hours 9:30 AM to 6:30 PM")])
add(S, "What time is the standard meal break?", "1:00 PM to 2:00 PM.", [(7, "Meal break 1:00 PM to 2:00 PM")])
add(S, "Until what time is the standard arrival grace period?", "Until 9:45 AM.", [(7, "The standard arrival grace period is until 9:45 AM")])
add(S, "How many late arrivals in a calendar month can trigger a manager discussion?", "More than five late arrivals in a calendar month.", [(7, "More than five late arrivals in a calendar month")])
add(S, "Within how many hours must an attendance correction request be submitted after discovering an error?", "Within 48 hours of discovering the error.", [(7, "Submit an attendance correction request within 48 hours of discovering an error")])
add(S, "After how many months of service can an employee request remote working days?", "After three months of service.", [(8, "completed three months of service may request up to two remote working days per week")])
add(S, "How many remote working days per week can an eligible employee request?", "Up to two remote working days per week.", [(8, "completed three months of service may request up to two remote working days per week")])
add(S, "What is the monthly internet reimbursement limit for eligible remote employees?", "Up to ₹1,000 per month, against required proof.", [(8, "may claim internet reimbursement up to ₹1,000 per month")])
add(S, "What is the annual CTC band for level L2 (Engineer)?", "₹6.0–10.0 LPA.", [(9, "L2 — Engineer Software Engineer, QA Engineer ₹6.0–10.0 LPA")])
add(S, "What is the annual CTC band for level L4 (Lead / Specialist)?", "₹16.0–24.0 LPA.", [(9, "L4 — Lead / Specialist Tech Lead, Security Specialist ₹16.0–24.0 LPA")])
add(S, "When is salary normally paid?", "On the last working day of the month.", [(10, "Salary payment Normally on the last working day of the month")])
add(S, "Within how many days must itemized expense receipts be submitted through the expense portal?", "Within 30 days.", [(10, "through the expense portal within 30 days")])
add(S, "Within how many working days of receiving the payslip should payroll queries be raised?", "Within 10 working days.", [(10, "within 10 working days of receiving the payslip")])
add(S, "What is the coverage limit of the group medical insurance?", "Up to ₹5,00,000, subject to insurer terms.", [(11, "Group medical insurance Coverage up to ₹5,00,000")])
add(S, "How much is the annual learning allowance for eligible full-time employees?", "Up to ₹12,000 per financial year, pre-approved.", [(11, "Learning allowance Up to ₹12,000 per financial year")])
add(S, "How many days of casual leave does a full-time employee get per year?", "8 days.", [(12, "Casual Leave (CL) 8 days")])
add(S, "How many days of earned leave does a full-time employee get per year?", "15 days.", [(12, "Earned Leave (EL) 15 days")])
add(S, "What is the maximum number of earned leave days that can be carried forward?", "A maximum of 30 earned leave days.", [(12, "A maximum of 30 earned leave days may be carried forward")])
add(S, "How many days of paternity leave are provided?", "10 working days.", [(12, "Paternity Leave 10 working days")])
add(S, "On what date is Republic Day in the 2027 holiday calendar?", "26 January 2027.", [(12, "26 January 2027 Republic Day National")])
add(S, "On what date is Diwali in the 2027 holiday calendar?", "29 October 2027.", [(13, "29 October 2027 Diwali Company")])
add(S, "How long is the standard probation period?", "Six months, unless the offer letter states otherwise.", [(15, "The standard probation period is six months")])
add(S, "How long is a standard internship?", "Three months.", [(16, "Standard internships are planned for three months")])
add(S, "What is the monthly stipend for interns?", "₹10,000 per month, subject to the signed agreement.", [(16, "Monthly stipend ₹10,000, subject to signed agreement")])
add(S, "How often are formal performance reviews conducted?", "Twice annually (January–June and July–December).", [(17, "Formal reviews are conducted twice annually")])
add(S, "What does a performance rating of 3 mean?", "Meets expectations: reliably fulfils the requirements of the role.", [(17, "3 Meets expectations Reliably fulfils the requirements of the role")])
add(S, "Within how many working days of receiving a formal review can an employee submit a written response to HR?", "Within 10 working days.", [(17, "submit a written response to HR within 10 working days of receiving the formal review")])
add(S, "Above what amount does a single training request need department-head approval?", "Above ₹15,000.", [(19, "A single training request above ₹15,000 requires department-head approval")])
add(S, "What is the minimum recommended password length?", "At least 12 characters where supported.", [(21, "Use unique passwords of at least 12 characters")])
add(S, "How soon must a lost or stolen company device be reported to IT/Security?", "Within one hour of discovery.", [(21, "within one hour of discovery")])
add(S, "What is the confidential grievance email address?", "grievance@asteriondigital.test.", [(22, "Confidential grievance channel: grievance@asteriondigital.test")])
add(S, "Within how many working days should ordinary grievances be acknowledged?", "Within three working days.", [(22, "Ordinary grievances should be acknowledged within three working days")])
add(S, "What is the notice period for senior employees?", "60 days.", [(23, "Senior employees 60 days")])
add(S, "Who is the Payroll Executive?", "Pooja Menon.", [(24, "Pooja Menon, Payroll Executive")])
add(S, "What is Vikram Reddy's designation?", "QA Engineer.", [(26, "Vikram Reddy QA Engineer Quality Assurance")])

# ============================ MULTI PART (25) ============================
add(M, "What are the company's six core values?",
    "Customer outcomes, Ownership, Craftsmanship, Integrity, Respect and Learning.",
    [(3, "Customer outcomes: understand the problem before proposing a solution"), (3, "Ownership: take responsibility for commitments"),
     (3, "Craftsmanship: write maintainable software"), (3, "Integrity: be honest about progress"),
     (3, "Respect: treat colleagues, clients and partners fairly"), (3, "Learning: share knowledge, seek feedback")])
add(M, "List the company's four offices with their headcounts.",
    "Bengaluru 260, Hyderabad 110, Pune 75 and Noida 55 (total 500).",
    [(5, "Bengaluru, Karnataka Head office, engineering, leadership and shared services 260"),
     (5, "Hyderabad, Telangana Cloud, data and enterprise delivery 110"),
     (5, "Pune, Maharashtra Product engineering and quality assurance 75"),
     (5, "Noida, Uttar Pradesh Client delivery, support and business operations 55")])
add(M, "Who are the CEO, COO, CFO and CISO of the company?",
    "CEO Aarav Sharma, COO Karan Malhotra, CFO Meera Iyer and CISO Sandeep Joshi.",
    [(5, "Chief Executive Officer (Aarav Sharma)"), (5, "Chief Operating Officer (Karan Malhotra)"),
     (5, "Chief Financial Officer (Meera Iyer)"), (5, "Chief Information Security Officer (Sandeep Joshi)")])
add(M, "What is the standard schedule: working days, core office hours, meal break and weekly rest days?",
    "Monday to Friday, core hours 9:30 AM to 6:30 PM, meal break 1:00 PM to 2:00 PM, weekly rest days Saturday and Sunday.",
    [(7, "Working days Monday to Friday"), (7, "Core office hours 9:30 AM to 6:30 PM"), (7, "Meal break 1:00 PM to 2:00 PM"),
     (7, "Weekly rest days Saturday and Sunday")])
add(M, "Under the flexible start time policy, what start window is allowed and what must employees still attend?",
    "With manager approval, employees may start between 8:00 AM and 11:00 AM; they must still attend mandatory customer meetings, team ceremonies and agreed collaboration windows.",
    [(7, "may start between 8:00 AM and 11:00 AM"), (7, "mandatory customer meetings, team ceremonies and agreed collaboration windows")])
add(M, "Which employees may be required to work onsite instead of remotely?",
    "New joiners, employees in structured training, employees handling restricted environments and roles requiring physical equipment access.",
    [(8, "New joiners, employees in structured training, employees handling restricted environments and roles requiring physical equipment access")])
add(M, "What does the manager confirm when approving a remote working day?",
    "Coverage, meetings, deliverables and any customer restrictions.",
    [(8, "The manager confirms coverage, meetings, deliverables and any customer restrictions")])
add(M, "Which components can make up an employee's CTC?",
    "Basic salary, House Rent Allowance, special/flexible allowance, employer PF contribution, gratuity provision, variable pay (only where stated) and insurance/benefits.",
    [(9, "Basic salary Fixed monthly component"), (9, "House Rent Allowance May be provided"), (9, "Special/flexible allowance Balancing fixed-pay component"),
     (9, "Employer PF contribution Included in CTC where applicable"), (9, "Gratuity provision Included in CTC"), (9, "Variable pay Only where expressly stated")])
add(M, "Describe the payroll cycle: payroll period, payment method, payslip and bank-holiday handling.",
    "The payroll period is the first to the last calendar day of each month; salary is paid by direct transfer to a verified bank account; payslips are published in the secure employee portal; if payday is a bank holiday, payment is normally processed on the preceding working day.",
    [(10, "Payroll period First to last calendar day of each month"), (10, "Payment method Direct transfer to verified employee bank account"),
     (10, "Payslip Published in the secure employee portal"), (10, "Bank holiday Payment is normally processed on the preceding working day")])
add(M, "What benefits are listed for eligible full-time employees?",
    "Group medical insurance (up to ₹5,00,000), personal accident cover, Provident Fund, gratuity, learning allowance (up to ₹12,000 per financial year), internet reimbursement (up to ₹1,000 per month) and annual health/wellness initiatives.",
    [(11, "Group medical insurance Coverage up to ₹5,00,000"), (11, "Personal accident cover"), (11, "Provident Fund As applicable under law"),
     (11, "Learning allowance Up to ₹12,000 per financial year"), (11, "Internet reimbursement Up to ₹1,000 per month"), (11, "Annual health / wellness initiatives")])
add(M, "How many days of casual, sick and earned leave and optional holidays does a full-time employee get each year?",
    "Casual leave 8 days, sick leave 8 days, earned leave 15 days and 2 optional holidays.",
    [(12, "Casual Leave (CL) 8 days"), (12, "Sick Leave (SL) 8 days"), (12, "Earned Leave (EL) 15 days"), (12, "Optional Holidays 2 days")])
add(M, "How far in advance should casual leave and earned leave be requested, and when should sick leave be reported?",
    "Casual leave one working day in advance, earned leave at least seven calendar days in advance, and sick leave as early as possible, preferably before 10:00 AM on the first day.",
    [(12, "Casual leave should normally be requested one working day in advance"), (12, "Earned leave should normally be requested at least seven calendar days in advance"),
     (12, "preferably before 10:00 AM on the first day")])
add(M, "What are the stages of the recruitment process?",
    "Role approval and job description, application screening, assessment, interview, reference or background checks, and offer and acceptance.",
    [(14, "Role approval and job description"), (14, "Application screening"), (14, "Assessment: technical, functional, writing or case-based exercises"),
     (14, "Interview: structured discussion"), (14, "Reference or background checks"), (14, "Offer and acceptance")])
add(M, "What activities are planned during the first 10 working days of onboarding?",
    "Day 1: welcome, identity verification, policies, equipment and access setup; Days 2–3: company overview, team introductions, security and conduct training; Days 4–5: role objectives, tools, architecture or process walkthrough; Week 2: starter tasks, mentor check-in, expectations and learning plan.",
    [(15, "Day 1 Welcome, identity verification, policies, equipment and access setup"), (15, "Days 2–3 Company overview, team introductions, security and conduct training"),
     (15, "Days 4–5 Role objectives, tools, architecture or process walkthrough"), (15, "Week 2 Starter tasks, mentor check-in, expectations and learning plan")])
add(M, "What are the key terms of the internship: duration, stipend, leave and completion evidence?",
    "Duration 3 months; stipend ₹10,000 per month; 2 casual leave days per month; certificate after successful completion and deliverable submission.",
    [(16, "Duration 3 months"), (16, "Monthly stipend ₹10,000, subject to signed agreement"), (16, "Leave 2 casual leave days per month, subject to agreement"),
     (16, "Certificate after successful completion and deliverable submission")])
add(M, "What dimensions are assessed in a performance review?",
    "Delivery & ownership, quality, technical/functional skill, collaboration, customer impact and professional conduct.",
    [(17, "Delivery & ownership Commitment reliability"), (17, "Quality Correctness, maintainability"), (17, "Technical / functional skill Role knowledge"),
     (17, "Collaboration Communication, teamwork"), (17, "Customer impact Responsiveness"), (17, "Professional conduct Integrity")])
add(M, "What are the five performance ratings and their labels?",
    "5 Exceptional, 4 Exceeds expectations, 3 Meets expectations, 2 Partially meets and 1 Does not meet.",
    [(17, "5 Exceptional"), (17, "4 Exceeds expectations"), (17, "3 Meets expectations"), (17, "2 Partially meets"), (17, "1 Does not meet")])
add(M, "What are the internal contribution bands used for talent discussions?",
    "Exceptional (top impact), Strong (above role expectations), Solid (meets expectations) and Improvement Required.",
    [(17, "Exceptional (top impact), Strong (above role expectations), Solid (meets expectations) and Improvement Required")])
add(M, "What does promotion readiness require?",
    "Consistent performance at the next-level scope, demonstrated judgment and accountability, evidence of collaboration/mentoring/leadership, a sustained business need with an approved position, and completion of any required assessment.",
    [(18, "Consistent performance at the next-level scope"), (18, "Demonstrated judgment, accountability"), (18, "Evidence of collaboration, mentoring or leadership"),
     (18, "A sustained business need and an approved position"), (18, "Completion of any required technical, compliance or leadership assessment")])
add(M, "What examples of conflicts of interest does the handbook give?",
    "Selecting a relative's business as a vendor, holding a material interest in a competitor, using confidential information for personal gain, and supervising a close relative without safeguards.",
    [(20, "selecting a relative's business as a vendor, holding a material interest in a competitor, using confidential information for personal gain or supervising a close relative without safeguards")])
add(M, "What are the four information classification levels?",
    "Public, Internal, Confidential and Restricted.",
    [(21, "Information may be classified as Public, Internal, Confidential or Restricted")])
add(M, "What reporting channels are available for harassment or grievances?",
    "The reporting manager, the People & Culture/HR representative, the designated Internal Committee (POSH), the confidential grievance channel, and emergency services or authorities for immediate safety risks.",
    [(22, "Reporting manager, unless the manager is involved"), (22, "People & Culture / HR representative"), (22, "Designated Internal Committee for complaints covered by POSH legislation"),
     (22, "Confidential grievance channel: grievance@asteriondigital.test"), (22, "Emergency services or appropriate authorities")])
add(M, "What are the notice periods for each employee category?",
    "Junior employees 30 days, mid-level 45 days, senior 60 days, department heads 90 days and interns 7 days (subject to agreement).",
    [(23, "Junior employees 30 days"), (23, "Mid-level employees 45 days"), (23, "Senior employees 60 days"), (23, "Department heads 90 days"), (23, "Interns 7 days, subject to agreement")])
add(M, "What must an employee do for handover when leaving the company?",
    "Prepare a status summary of projects, risks, dependencies and deadlines; transfer documentation, repositories and knowledge; return the laptop, access card, peripherals and other property; submit outstanding expense claims and resolve advances; complete security, finance and departmental clearances.",
    [(23, "Prepare a current status summary of projects, risks, dependencies and deadlines"), (23, "Transfer approved documentation, repositories and operational knowledge"),
     (23, "Return laptop, access card, peripherals and other company property"), (23, "Submit outstanding expense claims and resolve advances where applicable"),
     (23, "Complete required security, finance and departmental clearances")])
add(M, "Who are the contacts for People & Culture, Payroll & Finance and the IT Service Desk, and what are their email addresses?",
    "People & Culture: Deepa Nair, people@asteriondigital.test; Payroll & Finance: Pooja Menon, payroll@asteriondigital.test; IT Service Desk: Suresh Pillai, ithelp@asteriondigital.test.",
    [(24, "People & Culture Deepa Nair, Head of People & Culture people@asteriondigital.test"), (24, "Payroll & Finance Pooja Menon, Payroll Executive payroll@asteriondigital.test"),
     (24, "IT Service Desk Suresh Pillai, IT Support Lead ithelp@asteriondigital.test")])

# ============================ MULTI HOP (20) ============================
add(H, "An intern gets 2 casual leave days per month. How many casual leave days is that over a standard internship?",
    "A standard internship is 3 months, so an intern gets 6 casual leave days in total.",
    [(16, "Duration 3 months"), (16, "Leave 2 casual leave days per month, subject to agreement")])
add(H, "How much stipend does an intern earn over a full standard internship?",
    "₹10,000 per month for 3 months, which is ₹30,000.",
    [(16, "Duration 3 months"), (16, "Monthly stipend ₹10,000, subject to signed agreement")])
add(H, "An employee wants to enrol in a ₹16,000 certification. What approval is needed and how much can the learning allowance cover?",
    "A single request above ₹15,000 needs department-head approval, and the annual learning allowance covers up to ₹12,000.",
    [(19, "annual learning allowance of up to ₹12,000"), (19, "A single training request above ₹15,000 requires department-head approval")])
add(H, "Who approved the handbook, and which people in the employee directory report to that person?",
    "Karan Malhotra (COO, ADS010) approved it; Anil Kumar (Delivery Manager) and Suresh Pillai (IT Support Lead) report to him.",
    [(25, "Approved by Karan Malhotra, Chief Operating Officer"), (26, "ADS010 Karan Malhotra Chief Operating Officer Operations"),
     (26, "ADS015 Anil Kumar Delivery Manager Operations ADS010 Noida"), (27, "ADS018 Suresh Pillai IT Support Lead IT Services ADS010 Bengaluru")])
add(H, "Who does Neha Singh report to, and what is that person's designation?",
    "She reports to ADS004, Sneha Kapoor, who is the Engineering Manager.",
    [(26, "ADS006 Neha Singh Software Engineer Engineering ADS004 Pune"), (26, "ADS004 Sneha Kapoor Engineering Manager Engineering ADS002 Bengaluru")])
add(H, "Ananya Gupta is an intern in the directory. Who is her reporting manager, and what notice period applies to interns?",
    "Her manager is Ishita Bose (ADS013); the notice period for interns is 7 days, subject to agreement.",
    [(26, "ADS008 Ananya Gupta Data Analyst Intern Data & Analytics ADS013 Noida"), (26, "ADS013 Ishita Bose Manager, Data & Analytics Data & Analytics ADS002 Hyderabad"),
     (23, "Interns 7 days, subject to agreement")])
add(H, "Who is Rahul Mehta's reporting manager, and what is that person's email address?",
    "He reports to ADS009, Deepa Nair, whose email is deepa.nair@asteriondigital.test.",
    [(26, "ADS003 Rahul Mehta HR Manager Human Resources ADS009 Bengaluru"), (26, "ADS009 Deepa Nair Head of People & Culture Human Resources ADS001 Bengaluru 08 Jan 2018 deepa.nair@asteriondigital.test")])
add(H, "Who is the payroll contact, and who does that person report to according to the directory?",
    "Pooja Menon (Payroll Executive) reports to ADS011, Meera Iyer, the Chief Financial Officer.",
    [(24, "Pooja Menon, Payroll Executive"), (27, "ADS017 Pooja Menon Payroll Executive Finance ADS011 Bengaluru"), (26, "ADS011 Meera Iyer Chief Financial Officer Finance ADS001 Bengaluru")])
add(H, "What is the CTC band for an L3 Senior Engineer, and what is the example next career step?",
    "₹10.0–16.0 LPA, with L4 Lead / Specialist as the next step.",
    [(9, "L3 — Senior Engineer Senior Developer, Data Engineer ₹10.0–16.0 LPA"), (18, "L3 Senior Engineer Owns features or workstreams; mentors peers L4 Lead / Specialist")])
add(H, "An employee wants 12 consecutive working days of earned leave. How early should they request it and what extra review applies?",
    "Earned leave should be requested at least seven calendar days in advance, and leave exceeding 10 consecutive working days requires HR review.",
    [(12, "Earned leave should normally be requested at least seven calendar days in advance"),
     (12, "Leave exceeding 10 consecutive working days, leave without pay and extended medical leave require HR review")])
add(H, "A department head resigns. What notice period applies and what company property must be returned?",
    "The notice period is 90 days; the laptop, access card, peripherals and other company property must be returned.",
    [(23, "Department heads 90 days"), (23, "Return laptop, access card, peripherals and other company property")])
add(H, "How many days of casual, sick and earned leave plus optional holidays does a full-time employee get in total each year?",
    "8 + 8 + 15 + 2 = 33 days.",
    [(12, "Casual Leave (CL) 8 days"), (12, "Sick Leave (SL) 8 days"), (12, "Earned Leave (EL) 15 days"), (12, "Optional Holidays 2 days")])
add(H, "What is the most an eligible remote employee can claim for internet reimbursement in a year?",
    "Up to ₹1,000 per month, so ₹12,000 over 12 months.",
    [(8, "may claim internet reimbursement up to ₹1,000 per month"), (11, "Internet reimbursement Up to ₹1,000 per month for eligible remote employees")])
add(H, "Who owns policy interpretation, who leads that function, and what is its email address?",
    "Policy interpretation belongs to HR (People & Culture), led by Deepa Nair, at people@asteriondigital.test.",
    [(5, "Policy interpretation belongs to HR"), (24, "People & Culture Deepa Nair, Head of People & Culture people@asteriondigital.test")])
add(H, "Who is responsible for salary processing, who is the contact person and what is the email address?",
    "Salary processing belongs to Payroll/Finance; the contact is Pooja Menon at payroll@asteriondigital.test.",
    [(5, "salary processing belongs to Payroll/Finance"), (24, "Payroll & Finance Pooja Menon, Payroll Executive payroll@asteriondigital.test")])
add(H, "Which employees in the directory report to the Chief Technology Officer, Priya Verma?",
    "Sneha Kapoor, Rohan Das, Ishita Bose, Sandeep Joshi and Farhan Ali (all report to ADS002).",
    [(26, "ADS002 Priya Verma Chief Technology Officer Technology ADS001 Bengaluru"), (26, "ADS004 Sneha Kapoor Engineering Manager Engineering ADS002 Bengaluru"),
     (26, "ADS012 Rohan Das QA Manager Quality Assurance ADS002 Pune"), (26, "ADS013 Ishita Bose Manager, Data & Analytics Data & Analytics ADS002 Hyderabad"),
     (26, "ADS014 Sandeep Joshi Chief Information Security Officer Security ADS002 Bengaluru"), (27, "ADS016 Farhan Ali Cloud & DevOps Manager Cloud & Infrastructure ADS002 Hyderabad")])
add(H, "Which employees in the directory are based in Hyderabad?",
    "Vikram Reddy (QA Engineer), Ishita Bose (Manager, Data & Analytics) and Farhan Ali (Cloud & DevOps Manager).",
    [(26, "ADS007 Vikram Reddy QA Engineer Quality Assurance ADS012 Hyderabad"), (26, "ADS013 Ishita Bose Manager, Data & Analytics Data & Analytics ADS002 Hyderabad"),
     (27, "ADS016 Farhan Ali Cloud & DevOps Manager Cloud & Infrastructure ADS002 Hyderabad")])
add(H, "Where is Tanvi Deshpande based and who is her reporting manager?",
    "She is based in Pune and reports to ADS009, Deepa Nair.",
    [(27, "ADS019 Tanvi Deshpande Employee Relations Officer Human Resources ADS009 Pune"), (26, "ADS009 Deepa Nair Head of People & Culture Human Resources ADS001 Bengaluru")])
add(H, "Who handles privacy enquiries, and who does that person report to in the employee directory?",
    "Sandeep Joshi, the CISO, handles privacy enquiries; he reports to ADS002, Priya Verma (CTO).",
    [(24, "Privacy enquiries Sandeep Joshi, CISO privacy@asteriondigital.test"), (26, "ADS014 Sandeep Joshi Chief Information Security Officer Security ADS002 Bengaluru"),
     (26, "ADS002 Priya Verma Chief Technology Officer Technology ADS001 Bengaluru")])

add(H, "A new joiner has completed only two months of service. Can they request remote working days, and why?",
    "Not yet: remote working requires three months of service, and new joiners may be required to work onsite.",
    [(8, "completed three months of service may request up to two remote working days per week"), (8, "New joiners, employees in structured training")])

# ============================ UNANSWERABLE (20) ============================
NOT = "The handbook does not state this."
for q in [
    "What is the company's dress code?",
    "Does Asterion offer stock options or ESOPs to employees?",
    "What is the CEO's annual salary?",
    "What was the company's annual revenue last year?",
    "What is the monthly meal or food allowance?",
    "Does the company provide cab or transport facilities?",
    "What is the relocation allowance for employees transferred to another city?",
    "How many weeks of maternity leave does the company provide?",
    "What is the holiday calendar for 2028?",
    "Which insurance company provides the group medical policy?",
    "What is the phone number of the HR department?",
    "How many employees work in the Chennai office?",
    "What is Sneha Kapoor's salary?",
    "How many days of bereavement leave are allowed?",
    "What is the overtime pay rate per hour?",
    "Does the company offer a gym membership or sports club?",
    "On what date does the company's financial year end?",
    "How many days of marriage leave are allowed?",
    "Which cloud provider does Asterion use for its infrastructure?",
    "What is the company's policy on bringing pets to the office?",
]:
    add(U, q, NOT)

# ---------------------------------------------------------------------------
for i, row in enumerate(Q, 1):
    row["id"] = f"q{i:03d}"
    row["source"] = ["Asterion_Employee_Handbook.pdf"]

OUT.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in Q) + "\n", encoding="utf-8")
from collections import Counter
print(f"Wrote {len(Q)} questions to {OUT}")
print(dict(Counter(r["category"] for r in Q)))
