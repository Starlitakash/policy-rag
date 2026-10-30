"""
Generate realistic sample university policy PDFs for Shri Ram Institute of Technology and Research (SRITR).
These documents serve as dummy test data for learning and demonstrating the RAG pipeline.
"""

import os
from pathlib import Path
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, HRFlowable, PageBreak
from reportlab.lib import colors

DOCUMENTS_DIR = Path(__file__).resolve().parent.parent / "documents"
DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)

def create_policy_pdf(filename: str, title: str, doc_code: str, pages_data: list):
    """
    Creates a styled multi-page PDF document with realistic university policy headers and sections.
    """
    filepath = DOCUMENTS_DIR / filename
    doc = SimpleDocTemplate(
        str(filepath),
        pagesize=letter,
        rightMargin=54,
        leftMargin=54,
        topMargin=54,
        bottomMargin=54
    )

    styles = getSampleStyleSheet()

    # Custom styles
    header_title_style = ParagraphStyle(
        'UnivHeader',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=14,
        leading=18,
        alignment=1, # Center
        textColor=colors.HexColor('#1E3A8A')
    )
    header_sub_style = ParagraphStyle(
        'UnivSub',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10,
        leading=13,
        alignment=1,
        textColor=colors.HexColor('#4B5563')
    )
    doc_title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=16,
        leading=20,
        alignment=1,
        textColor=colors.HexColor('#0F172A'),
        spaceAfter=10
    )
    meta_style = ParagraphStyle(
        'DocMeta',
        parent=styles['Normal'],
        fontName='Helvetica-Oblique',
        fontSize=9,
        leading=12,
        alignment=1,
        textColor=colors.HexColor('#6B7280'),
        spaceAfter=15
    )
    section_heading_style = ParagraphStyle(
        'SectionHeading',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=12,
        leading=16,
        textColor=colors.HexColor('#1E40AF'),
        spaceBefore=12,
        spaceAfter=6
    )
    body_style = ParagraphStyle(
        'PolicyBody',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10,
        leading=14,
        textColor=colors.HexColor('#1F2937'),
        spaceAfter=8
    )
    callout_style = ParagraphStyle(
        'Callout',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=9.5,
        leading=13,
        textColor=colors.HexColor('#991B1B'),
        backColor=colors.HexColor('#FEE2E2'),
        borderPadding=6,
        spaceBefore=6,
        spaceAfter=10
    )

    story = []

    for page_idx, page_sections in enumerate(pages_data):
        # Header on every page
        story.append(Paragraph("SHRI RAM INSTITUTE OF TECHNOLOGY AND RESEARCH (SRITR)", header_title_style))
        story.append(Paragraph("Knowledge Park III, Greater Noida, UP - 201306 | Office of Academic Affairs", header_sub_style))
        story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor('#1E3A8A'), spaceBefore=6, spaceAfter=10))

        if page_idx == 0:
            story.append(Paragraph(title, doc_title_style))
            story.append(Paragraph(f"Document Code: {doc_code} | Effective Academic Year: 2025-2026 | Version: 3.2", meta_style))
            story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor('#9CA3AF'), spaceBefore=2, spaceAfter=12))

        for item in page_sections:
            if item.get("type") == "heading":
                story.append(Paragraph(item["text"], section_heading_style))
            elif item.get("type") == "body":
                story.append(Paragraph(item["text"], body_style))
            elif item.get("type") == "callout":
                story.append(Paragraph(item["text"], callout_style))

        if page_idx < len(pages_data) - 1:
            story.append(PageBreak())

    doc.build(story)
    print(f"[OK] Generated {filename}")

def generate_all_dummy_policies():
    """Generates a complete suite of realistic policy documents for SRITR."""

    # 1. Academic Attendance Policy
    create_policy_pdf(
        filename="SRITR_Academic_Attendance_Policy.pdf",
        title="ACADEMIC ATTENDANCE & CONDONATION POLICY",
        doc_code="SRITR/POL/ACAD/2025/01",
        pages_data=[
            [
                {"type": "heading", "text": "1. Mandatory Attendance Requirements"},
                {"type": "body", "text": "1.1 Every registered student at Shri Ram Institute of Technology and Research (SRITR) is required to maintain a minimum of 75% attendance in each registered theory course, tutorial, and laboratory session to be eligible to appear in the End-Semester Examinations (ESE)."},
                {"type": "body", "text": "1.2 Attendance is computed on a per-course basis from the first official day of teaching until the last instructional working day declared in the university academic calendar."},
                {"type": "callout", "text": "CRITICAL RULE: Students with overall attendance falling below 65% under any circumstances will NOT be eligible for medical condonation and shall be summarily debarred."},
                {"type": "heading", "text": "2. Medical Leave & Attendance Condonation"},
                {"type": "body", "text": "2.1 The University Academic Council may grant condonation of attendance up to a maximum of 10% (lowering the threshold from 75% to 65%) strictly on valid medical grounds."},
                {"type": "body", "text": "2.2 To claim medical condonation, the student must submit an official Medical Certificate issued or counter-signed by the SRITR Chief Medical Officer (CMO) within seven (7) working days of returning to campus."},
            ],
            [
                {"type": "heading", "text": "3. Official Duty (OD) Leave & Co-Curricular Representation"},
                {"type": "body", "text": "3.1 Students representing SRITR in authorized inter-university sports, national hackathons, technical conferences, or NCC/NSS camps are entitled to On-Duty (OD) attendance credit."},
                {"type": "body", "text": "3.2 OD leave is capped at a maximum of fourteen (14) working days per semester. Prior written recommendation of the Faculty In-Charge and formal approval of the Dean of Student Welfare is mandatory before leaving campus."},
                {"type": "heading", "text": "4. Consequences of Attendance Shortage & Debarment"},
                {"type": "body", "text": "4.1 Students who are debarred due to shortage of attendance in any course will receive an 'FA' (Fail due to Attendance) grade on their official grade card."},
                {"type": "body", "text": "4.2 Debarred students must re-register for the course during the Summer Remedial Semester by paying the prescribed course registration fee of Rs. 4,500 per credit."},
            ]
        ]
    )

    # 2. Examination and Re-evaluation Policy
    create_policy_pdf(
        filename="SRITR_Examination_and_Re-evaluation_Policy.pdf",
        title="EXAMINATION, GRADING & RE-EVALUATION REGULATIONS",
        doc_code="SRITR/POL/EXAM/2025/02",
        pages_data=[
            [
                {"type": "heading", "text": "1. Evaluation Scheme & Weightage"},
                {"type": "body", "text": "1.1 The academic performance in every theory subject is evaluated through Continuous Internal Assessment (CIA) carrying 40% weightage and End-Semester Examination (ESE) carrying 60% weightage."},
                {"type": "body", "text": "1.2 CIA comprises two Mid-Term Exams (20%), Surprise Quizzes/Assignments (10%), and Class Participation & Regularity (10%)."},
                {"type": "heading", "text": "2. Minimum Passing Thresholds"},
                {"type": "body", "text": "2.1 To pass a course, a student must secure a minimum of 40% marks in the End-Semester Examination (24 out of 60) and an overall aggregate of 50% marks in CIA and ESE combined."},
                {"type": "callout", "text": "NOTE: Securing 50% overall but failing to score 40% in the final exam results in an 'F' grade."},
            ],
            [
                {"type": "heading", "text": "3. Answer Script Viewing & Re-evaluation Procedure"},
                {"type": "body", "text": "3.1 Stage 1 - Script Inspection: A candidate can apply for a digital photocopy of their evaluated answer script within seven (7) calendar days of result declaration upon payment of a non-refundable fee of Rs. 300 per subject."},
                {"type": "body", "text": "3.2 Stage 2 - Formal Re-evaluation: If discrepancies or evaluation errors exist, an application for formal re-evaluation by an external expert examiner must be submitted within fifteen (15) calendar days with a fee of Rs. 1,000 per subject."},
                {"type": "body", "text": "3.3 Fee Refund Clause: If the re-evaluation results in a mark increase of 10% or more of maximum theory marks, 50% of the re-evaluation fee (Rs. 500) will be credited back to the student's account."},
                {"type": "heading", "text": "4. Supplementary & Backlog Examinations"},
                {"type": "body", "text": "4.1 Backlog examinations are held twice a year: Supplementary Exams in July and Winter Special Exams in January."},
                {"type": "body", "text": "4.2 A student is permitted to carry forward a maximum of four (4) uncleared backlogs into the final year (7th Semester)."},
            ]
        ]
    )

    # 3. Hostel and Leave Rules
    create_policy_pdf(
        filename="SRITR_Hostel_and_Leave_Rules.pdf",
        title="CAMPUS RESIDENCE & HOSTEL LEAVE REGULATIONS",
        doc_code="SRITR/POL/HOST/2025/03",
        pages_data=[
            [
                {"type": "heading", "text": "1. Hostel In-Time & Curfew Policy"},
                {"type": "body", "text": "1.1 All student residents must return to their respective hostel premises by 9:30 PM on all days of the week."},
                {"type": "body", "text": "1.2 Grace entry is permitted between 9:30 PM and 10:00 PM subject to biometric logging and written remarks in the Late Register at the security gate."},
                {"type": "callout", "text": "DISCIPLINE CLAUSE: More than three (3) late entries in a single calendar month attracts a mandatory fine of Rs. 500 and triggers an automated intimation email/SMS to parents."},
                {"type": "heading", "text": "2. Night Out & Weekend Leave Permission"},
                {"type": "body", "text": "2.1 Any resident student planning to stay overnight outside the hostel campus must apply for an Out-Pass on the SRITR Hostel Portal at least 48 hours in advance."},
                {"type": "body", "text": "2.2 The leave request is approved only after One-Time Password (OTP) verification and explicit consent from the parent/guardian's registered phone number."},
            ],
            [
                {"type": "heading", "text": "3. Prohibited Electrical Appliances & Safety"},
                {"type": "body", "text": "3.1 For electrical fire safety, heating rods, electric kettles, induction hotplates, room heaters, and high-wattage air conditioners are strictly forbidden in student rooms."},
                {"type": "body", "text": "3.2 Confiscated unauthorized appliances will not be returned until the end of the academic year, and the occupant will be fined Rs. 2,000 for the first offense."},
                {"type": "heading", "text": "4. Visitor & Day Scholar Regulations"},
                {"type": "body", "text": "4.1 Day scholars and non-resident visitors are strictly prohibited from entering student rooms at any time."},
                {"type": "body", "text": "4.2 Parents and authorized guests may meet residents only in the Central Visitor Lounge between 4:00 PM and 7:00 PM on Saturdays and Sundays."},
            ]
        ]
    )

    # 4. Library and Resource Policy
    create_policy_pdf(
        filename="SRITR_Library_and_Resource_Policy.pdf",
        title="CENTRAL LIBRARY CODE OF CONDUCT & RESOURCE CIRCULATION",
        doc_code="SRITR/POL/LIB/2025/04",
        pages_data=[
            [
                {"type": "heading", "text": "1. Library Operating Hours"},
                {"type": "body", "text": "1.1 The SRITR Central Library remains open on all academic working days from 8:00 AM to 11:00 PM. During examination periods, the reading halls remain operational 24/7."},
                {"type": "body", "text": "1.2 On Sundays and public holidays, the library operates on reduced hours from 9:00 AM to 6:00 PM."},
                {"type": "heading", "text": "2. Borrowing Privileges & Issue Limits"},
                {"type": "body", "text": "2.1 Undergraduate (B.Tech) Students: May borrow up to 4 books for a duration of 14 calendar days."},
                {"type": "body", "text": "2.2 Postgraduate (M.Tech/MBA) Students: May borrow up to 6 books for a duration of 28 calendar days."},
                {"type": "body", "text": "2.3 Research Scholars (Ph.D.): May borrow up to 8 books for a duration of 45 calendar days."},
            ],
            [
                {"type": "heading", "text": "3. Overdue Fines & Penalty Schedule"},
                {"type": "body", "text": "3.1 Books returned after the due date incur an overdue fine of Rs. 5 per day per book for the first seven (7) overdue days."},
                {"type": "body", "text": "3.2 If the delay exceeds seven days, the fine escalates to Rs. 15 per day per book from the 8th day until the book is returned or reported lost."},
                {"type": "heading", "text": "4. Lost or Damaged Book Replacement Policy"},
                {"type": "body", "text": "4.1 If a borrower loses a library book, they must either replace the book with a brand new copy of the same or newer edition plus pay a processing fee of Rs. 200."},
                {"type": "body", "text": "4.2 If the book is out of print, the borrower must pay three times (3x) the latest catalogue price of the book."},
            ]
        ]
    )

    # 5. Student Discipline and Anti-Ragging Code
    create_policy_pdf(
        filename="SRITR_Student_Discipline_and_Anti-Ragging_Code.pdf",
        title="STUDENT CODE OF CONDUCT & ZERO-TOLERANCE ANTI-RAGGING POLICY",
        doc_code="SRITR/POL/DISC/2025/05",
        pages_data=[
            [
                {"type": "heading", "text": "1. Zero Tolerance on Ragging (Statutory Mandate)"},
                {"type": "body", "text": "1.1 In compliance with UGC Regulations and the Hon'ble Supreme Court directives, Shri Ram Institute of Technology and Research (SRITR) enforces a ZERO TOLERANCE policy against ragging in any form."},
                {"type": "body", "text": "1.2 Ragging includes any conduct by spoken words, written act, or physical abuse that causes teasing, embarrassment, physical hurt, psychological harm, or raises apprehension in a fresher or fellow student."},
                {"type": "callout", "text": "STATUTORY NOTICE: Ragging is a non-bailable criminal offense. Perpetrators face immediate expulsion, forfeiture of degree credits, and criminal FIR under IPC sections."},
                {"type": "heading", "text": "2. 24x7 Anti-Ragging Flying Squad & Emergency Helpline"},
                {"type": "body", "text": "2.1 Toll-Free Campus Emergency Helpline: 1800-180-5522 | Anti-Ragging Email: antiragging@sritr.ac.in"},
                {"type": "body", "text": "2.2 The Proctorial Flying Squad conducts unannounced 24-hour patrols across hostels, canteens, bus stops, and academic zones."},
            ],
            [
                {"type": "heading", "text": "3. Disciplinary Penalties for Misconduct"},
                {"type": "body", "text": "3.1 Any student found guilty of campus vandalism, substance abuse, cyber-bullying, or academic cheating faces the following graduated sanctions:"},
                {"type": "body", "text": "• Level 1 Sanction: Official written reprimand and community service (15 hours)."},
                {"type": "body", "text": "• Level 2 Sanction: Suspension from classes and hostel for one (1) to two (2) semesters."},
                {"type": "body", "text": "• Level 3 Sanction: Permanent rustication and cancellation of university registration."},
                {"type": "heading", "text": "4. Mandatory Campus Dress Code & ID Card Display"},
                {"type": "body", "text": "4.1 All students must wear their official SRITR Smart ID Card on a university lanyard visibly at all times when inside campus gates."},
                {"type": "body", "text": "4.2 Formal or smart-casual attire is mandatory in academic lecture halls and laboratories; flip-flops, bathroom slippers, and sleepwear are strictly prohibited."},
            ]
        ]
    )

    # 6. Scholarship and Fee Payment Policy
    create_policy_pdf(
        filename="SRITR_Scholarship_and_Fee_Payment_Policy.pdf",
        title="STUDENT SCHOLARSHIPS, TUITION GRANTS & FEE REFUND POLICY",
        doc_code="SRITR/POL/FEES/2025/06",
        pages_data=[
            [
                {"type": "heading", "text": "1. Merit-cum-Means Tuition Fee Waivers"},
                {"type": "body", "text": "1.1 Category A (SRITR Chancellor's Scholar): 100% tuition fee waiver awarded to top 5 rankers in university entrance with parental annual income less than Rs. 5.00 Lakhs."},
                {"type": "body", "text": "1.2 Category B (Academic Excellence Grant): 50% tuition fee waiver for continuing students maintaining a Semester Grade Point Average (SGPA) >= 9.25 with no backlog history."},
                {"type": "heading", "text": "2. Semester Fee Payment Schedule & Late Surcharge"},
                {"type": "body", "text": "2.1 Semester tuition fees must be cleared within fifteen (15) calendar days from the opening date of each academic semester."},
                {"type": "body", "text": "2.2 Late payments between day 16 and day 30 attract a late fee of Rs. 100 per day. If fees remain unpaid after 30 days, ERP portal access is frozen."},
            ],
            [
                {"type": "heading", "text": "3. Admission Withdrawal & Fee Refund Matrix"},
                {"type": "body", "text": "3.1 Withdrawal submitted 15 or more days before formal orientation date: 100% refund of aggregate tuition fees minus Rs. 1,000 administrative processing deduction."},
                {"type": "body", "text": "3.2 Withdrawal submitted less than 15 days before orientation date: 80% refund of aggregate fees."},
                {"type": "body", "text": "3.3 Withdrawal submitted up to 15 days after orientation date: 50% refund of aggregate fees."},
                {"type": "body", "text": "3.4 Withdrawal submitted more than 30 days after class commencement: 0% refund (only refundable security deposit refunded)."},
            ]
        ]
    )

if __name__ == "__main__":
    generate_all_dummy_policies()
    print("\nAll dummy policy documents generated successfully in documents/ folder!")
