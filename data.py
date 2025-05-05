import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import re
import string
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight


class SentimentDataset(Dataset):
    def __init__(self, texts, contexts, labels, word_to_idx, max_len_text=50, max_len_context=20):
        self.texts = texts
        self.contexts = contexts
        self.labels = labels
        self.word_to_idx = word_to_idx
        self.max_len_text = max_len_text
        self.max_len_context = max_len_context

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        context = self.contexts[idx]
        label = self.labels[idx]

        # Convert text and context to indices
        text_indices = [self.word_to_idx.get(word, self.word_to_idx['<UNK>']) for word in text.split()]
        context_indices = [self.word_to_idx.get(word, self.word_to_idx['<UNK>']) for word in context.split()]

        # Padding or truncating
        if len(text_indices) < self.max_len_text:
            text_indices += [self.word_to_idx['<PAD>']] * (self.max_len_text - len(text_indices))
        else:
            text_indices = text_indices[:self.max_len_text]

        if len(context_indices) < self.max_len_context:
            context_indices += [self.word_to_idx['<PAD>']] * (self.max_len_context - len(context_indices))
        else:
            context_indices = context_indices[:self.max_len_context]

        return {
            'text': torch.tensor(text_indices, dtype=torch.long),
            'context': torch.tensor(context_indices, dtype=torch.long),
            'label': torch.tensor(label, dtype=torch.long)
        }


def preprocess_text(text, remove_stopwords=False):
    """
    Simple preprocessing: lowercase, remove punctuation.
    No NLTK dependency to avoid installation issues.
    """
    # Convert to lowercase
    text = text.lower()

    # Remove punctuation and replace with space
    text = re.sub(r'[^\w\s]', ' ', text)

    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def build_vocabulary(texts, contexts, max_vocab_size=5000):
    """Build a vocabulary from all texts and contexts."""
    all_words = []
    for text in texts:
        all_words.extend(text.split())
    for context in contexts:
        all_words.extend(context.split())

    # Count word frequencies
    word_counts = Counter(all_words)

    # Sort by frequency and limit to max_vocab_size
    most_common = word_counts.most_common(max_vocab_size - 3)  # Reserve space for <PAD>, <UNK>, etc.

    # Create word to index mapping
    word_to_idx = {'<PAD>': 0, '<UNK>': 1, '<BOS>': 2}  # Add beginning of sentence token
    for word, _ in most_common:
        word_to_idx[word] = len(word_to_idx)

    return word_to_idx


def load_data(csv_path, max_vocab_size=5000, batch_size=32, test_size=0.2,
              remove_stopwords=False, random_state=42):
    """Load data from CSV, preprocess, and create DataLoaders."""
    df = pd.read_csv(csv_path)

    # Check required columns are present
    if not all(col in df.columns for col in ['text', 'context', 'label']):
        raise ValueError("CSV file must contain 'text', 'context', and 'label' columns")

    # Map text labels to integers if needed
    if not pd.api.types.is_numeric_dtype(df['label']):
        label_map = {'Positive': 0, 'Negative': 1, 'Neutral': 2}
        df['label'] = df['label'].map(label_map)

    # Preprocess texts and contexts - ignoring remove_stopwords option to avoid NLTK dependency
    df['text'] = df['text'].apply(preprocess_text)
    df['context'] = df['context'].apply(preprocess_text)

    # Handle missing values
    df = df.dropna()

    # Split into train and test sets
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df['label']
    )

    # Build vocabulary from training data only
    word_to_idx = build_vocabulary(train_df['text'], train_df['context'], max_vocab_size)

    # Create datasets
    train_dataset = SentimentDataset(
        train_df['text'].values,
        train_df['context'].values,
        train_df['label'].values,
        word_to_idx
    )

    test_dataset = SentimentDataset(
        test_df['text'].values,
        test_df['context'].values,
        test_df['label'].values,
        word_to_idx
    )

    # Calculate class weights for handling imbalance
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(train_df['label']),
        y=train_df['label']
    )
    class_weights = torch.tensor(class_weights, dtype=torch.float)

    # Create DataLoaders with safer options for various environments
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0  # Avoid multiprocessing issues
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0  # Avoid multiprocessing issues
    )

    return train_loader, test_loader, word_to_idx, class_weights


import pandas as pd
import numpy as np
import random
from datetime import datetime
import os

# Đặt seed để tái lập
random.seed(42)
np.random.seed(42)


def generate_sample_data(output_path='sentiment_data.csv', num_samples=650):
    """
    Tạo bộ dữ liệu đa dạng cho phân tích cảm xúc với văn bản và ngữ cảnh

    Args:
        output_path: Đường dẫn lưu file CSV
        num_samples: Số lượng mẫu (tạo thêm để đảm bảo ít nhất 500 mẫu sau khi lọc)

    Returns:
        str: Đường dẫn đến file dữ liệu đã tạo
    """
    # =========== 1. Công việc & Môi trường làm việc ===========
    work_texts_positive = [
        "The new project management system streamlined our workflow significantly.",
        "Our team exceeded the quarterly targets by focusing on customer retention.",
        "The collaborative approach to problem-solving has improved productivity.",
        "My performance review highlighted my strengths in leadership and innovation.",
        "The company's flexible work policy gives me a better work-life balance.",
        "My manager recognized my contributions during our monthly team meeting.",
        "The office redesign created a more inspiring workspace for creativity.",
        "Our department successfully implemented the new efficiency protocols.",
        "The feedback from stakeholders on our latest project was overwhelmingly positive.",
        "The professional development workshop taught me valuable communication skills.",
        "My promotion came with responsibilities that align perfectly with my career goals.",
        "The company retreat strengthened team bonds across departments.",
        "Our client commended the quality and timeliness of our deliverables.",
        "The new hiring process brought exceptional talent to our team.",
        "The mentorship program accelerated my professional growth this year."
    ]

    work_contexts_positive = [
        "After months of planning.",
        "Despite initial resource constraints.",
        "Through consistent team effort.",
        "Following the leadership transition.",
        "During the company restructuring.",
        "With minimal outside consultation.",
        "In our competitive industry.",
        "Under tight deadline pressure.",
        "While adapting to market changes.",
        "Without increasing the budget."
    ]

    work_texts_negative = [
        "The new performance metrics fail to account for quality over quantity.",
        "My supervisor took credit for my ideas during the executive presentation.",
        "The mandatory overtime is causing significant team burnout.",
        "Budget cuts have eliminated essential tools for our daily operations.",
        "The office relocation doubled my commute time and expenses.",
        "Constant interruptions from other departments disrupt our workflow.",
        "The new reporting structure created unnecessary bureaucracy.",
        "Internal communication has deteriorated since the management change.",
        "The updated policies restrict our autonomy in decision-making.",
        "Our suggestions for improvement are consistently ignored by leadership.",
        "The last-minute project changes required weekend work without compensation.",
        "The promised resources for our department never materialized.",
        "Team morale has declined due to lack of recognition and support.",
        "The micromanagement style of our new director stifles creativity.",
        "The merger resulted in redundant roles and unclear responsibilities."
    ]

    work_contexts_negative = [
        "Despite multiple team complaints.",
        "Following the third reorganization.",
        "After promising otherwise.",
        "Without consulting affected teams.",
        "Contrary to industry standards.",
        "During our busiest season.",
        "Ignoring expert recommendations.",
        "After years of successful approaches.",
        "With no transition period.",
        "Breaking previous commitments."
    ]

    work_texts_neutral = [
        "The quarterly report indicates stable performance across departments.",
        "The office will remain open during standard business hours next week.",
        "Team assignments will be distributed according to expertise and availability.",
        "The client requested minor adjustments to the initial proposal.",
        "Performance reviews are scheduled for the last week of the month.",
        "The new system requires the same amount of data entry as before.",
        "Department budgets remain unchanged for the upcoming fiscal year.",
        "The conference room is available for booking through the online portal.",
        "Project timelines have been adjusted to accommodate the holiday schedule.",
        "The training materials cover both old and new operational procedures.",
        "Employee benefits packages will be reviewed by HR during annual enrollment.",
        "The company directory has been updated with recent staffing changes.",
        "Parking arrangements will continue as previously established.",
        "The survey results showed mixed opinions on the cafeteria options.",
        "Workstation equipment meets but doesn't exceed industry standards."
    ]

    work_contexts_neutral = [
        "As stated in the company memo.",
        "According to established protocol.",
        "As previously scheduled.",
        "Per standard procedures.",
        "Following quarterly assessment.",
        "Based on current regulations.",
        "As documented in the handbook.",
        "During regular business hours.",
        "In line with department policy.",
        "As mentioned in orientation."
    ]

    # =========== 2. Đánh giá sản phẩm ===========
    product_texts_positive = [
        "This laptop's battery lasts all day, even with intensive video editing work.",
        "The noise-cancelling headphones completely transformed my daily commute experience.",
        "The ergonomic chair eliminated my back pain after long hours of sitting.",
        "This smartphone camera captures incredible detail even in low light conditions.",
        "The waterproof hiking boots kept my feet dry crossing multiple streams.",
        "This kitchen knife maintains its sharpness even after months of daily use.",
        "The air purifier noticeably improved my allergies within just three days.",
        "This e-reader's anti-glare screen allows comfortable reading in direct sunlight.",
        "The robot vacuum navigates around furniture without getting stuck.",
        "This moisturizer restored my skin's hydration without causing breakouts.",
        "The wireless charger works perfectly even through my phone's thick case.",
        "This cast iron pan distributes heat evenly for perfect cooking results.",
        "The smart thermostat reduced my energy bill while maintaining comfort.",
        "This coffee grinder produces consistent grounds for exceptional flavor.",
        "The anti-fog swimming goggles stayed clear throughout my two-hour swim."
    ]

    product_contexts_positive = [
        "Compared to my previous model.",
        "Even with daily intensive use.",
        "After thorough research.",
        "Worth every penny spent.",
        "Beyond my expectations.",
        "Better than higher-priced alternatives.",
        "Despite initial skepticism.",
        "After trying many competitors.",
        "Even in challenging conditions.",
        "Exactly as advertised."
    ]

    product_texts_negative = [
        "The blender broke after only three uses despite its premium price tag.",
        "This smartphone battery drains completely within four hours of normal use.",
        "The wireless earbuds disconnect constantly during my daily workout.",
        "This winter jacket lets in cold air through poorly sealed seams.",
        "The expensive face cream caused an immediate allergic reaction.",
        "This fitness tracker reports wildly inaccurate heart rate measurements.",
        "The laptop overheats to uncomfortable levels during video calls.",
        "This car accessory damaged my vehicle's interior during installation.",
        "The supposed non-stick pan requires excessive oil to prevent food sticking.",
        "This printer consumes ink cartridges at an alarming and costly rate.",
        "The smart home device frequently misunderstands basic voice commands.",
        "This gaming controller developed stick drift within weeks of purchase.",
        "The water filter leaks from the bottom, creating a mess on my counter.",
        "This luggage wheel broke during its first trip through the airport.",
        "The supposedly durable phone case cracked from a minor drop."
    ]

    product_contexts_negative = [
        "Despite the premium pricing.",
        "Contrary to advertised features.",
        "Unlike my previous version.",
        "After just minimal usage.",
        "While following care instructions.",
        "Even with gentle handling.",
        "Against manufacturer claims.",
        "Following recommended settings.",
        "Beyond the return period.",
        "Without any misuse."
    ]

    product_texts_neutral = [
        "The product dimensions match the specifications listed on the website.",
        "This device requires standard AA batteries not included in the package.",
        "The clothing item fits according to the provided size chart measurements.",
        "This appliance operates at standard voltage compatible with most homes.",
        "The assembly requires a Phillips screwdriver and approximately 30 minutes.",
        "This software works with both Windows and Mac operating systems.",
        "The product color appears slightly different than in online photos.",
        "This item ships in the manufacturer's original packaging without gift wrapping.",
        "The warranty covers manufacturing defects for one year after purchase.",
        "This device includes the same features as the previous model.",
        "The product requires registration through the company website.",
        "This furniture needs to be cleaned with specific recommended products.",
        "The instruction manual comes in multiple languages including English.",
        "This subscription auto-renews at the current market rate.",
        "The product packaging is made from partially recycled materials."
    ]

    product_contexts_neutral = [
        "According to the manual.",
        "As listed in specifications.",
        "Similar to comparable models.",
        "Based on manufacturer information.",
        "Per industry standards.",
        "Common for this product category.",
        "As shown in product diagrams.",
        "Typical for this price point.",
        "Standard for this brand.",
        "As expected for this item."
    ]

    # =========== 3. Trải nghiệm dịch vụ ===========
    service_texts_positive = [
        "The hotel staff upgraded our room without any additional charges.",
        "The technician diagnosed and fixed the issue within the first hour.",
        "Our server remembered our preferences from our previous visit.",
        "The online customer service resolved my issue with just one message.",
        "The airline staff handled the delay with exceptional communication.",
        "The hairstylist listened carefully to create exactly what I wanted.",
        "The food delivery arrived hot and earlier than the estimated time.",
        "The tour guide customized our experience based on our interests.",
        "The dental office made my anxious child feel completely comfortable.",
        "The bank representative found a solution that saved me money.",
        "The car service included a thorough inspection at no extra cost.",
        "The event planner handled last-minute changes with calm professionalism.",
        "The fitness instructor modified exercises to accommodate my injury.",
        "The veterinarian took extra time to explain all treatment options.",
        "The insurance agent helped find coverage perfectly tailored to my needs."
    ]

    service_contexts_positive = [
        "During peak holiday season.",
        "Without requiring a reservation.",
        "Despite being extremely busy.",
        "Going above industry standards.",
        "With genuine personal care.",
        "Exceeding typical service levels.",
        "Making me feel valued.",
        "With exceptional expertise.",
        "Solving a complex problem.",
        "Creating repeat customer loyalty."
    ]

    service_texts_negative = [
        "The restaurant served cold food after we waited over an hour.",
        "The repair service left without fixing the original problem.",
        "Our hotel room was nothing like the photographs on the website.",
        "The customer service representative refused to escalate my complaint.",
        "The salon completely ignored my styling instructions.",
        "The delivery service left packages in the rain despite clear instructions.",
        "The tour operator packed twice the advertised number of people on the bus.",
        "The carpenter left sawdust and debris throughout my house.",
        "The internet installation technician missed three scheduled appointments.",
        "The medical office billed me for services that were supposed to be covered.",
        "The moving company damaged furniture and denied responsibility.",
        "The travel agency booked us into a hotel that was under construction.",
        "The landscaper killed healthy plants while removing weeds.",
        "The consultant provided generic advice despite charging premium rates.",
        "The property manager ignored repeated maintenance requests for months."
    ]

    service_contexts_negative = [
        "Despite being a loyal customer.",
        "After explicit instructions.",
        "Completely unapologetically.",
        "Charging premium prices.",
        "With no explanation offered.",
        "Breaking promises made.",
        "Without any prior notice.",
        "After confirming arrangements.",
        "With dismissive attitude.",
        "While ignoring feedback."
    ]

    service_texts_neutral = [
        "The restaurant requires reservations for parties larger than six people.",
        "The repair service operates during regular business hours on weekdays.",
        "Hotel checkout time is at noon, with luggage storage available afterward.",
        "The subscription service bills monthly with an option for annual payment.",
        "The transportation service follows a set route with designated stops.",
        "The salon accepts both walk-ins and appointments based on availability.",
        "The delivery window falls between 2PM and 6PM on the selected date.",
        "The tour includes transportation but meals must be purchased separately.",
        "The medical office requires insurance information prior to appointments.",
        "The maintenance service occurs quarterly according to the contract.",
        "The library lending period extends to three weeks with one renewal option.",
        "The cleaning service brings their own supplies as part of the standard package.",
        "The internet provider charges a one-time installation fee for new accounts.",
        "The training program consists of six weekly sessions of one hour each.",
        "The warranty service requires original proof of purchase documentation."
    ]

    service_contexts_neutral = [
        "As stated in their policy.",
        "Standard for this industry.",
        "According to their website.",
        "Typical for this service type.",
        "As mentioned during booking.",
        "Per their terms of service.",
        "Following normal procedures.",
        "Like most service providers.",
        "As explained during consultation.",
        "Based on their business model."
    ]

    # =========== 4. Tương tác xã hội ===========
    social_texts_positive = [
        "My friend drove two hours to help me move on short notice.",
        "The community organized a meal train when my family was in crisis.",
        "My colleague covered my shifts while I recovered from surgery.",
        "The neighbors welcomed us with homemade food when we moved in.",
        "Old friends picked up our conversation as if no time had passed.",
        "The study group stayed late to help me understand difficult concepts.",
        "My roommate surprised me with a celebration after my job interview.",
        "The online community raised funds for my medical expenses.",
        "My mentor made time for regular check-ins during my transition.",
        "Strangers helped push my car when it broke down in traffic.",
        "The party host made special accommodations for my dietary restrictions.",
        "My siblings organized a perfect surprise for our parents' anniversary.",
        "The community garden group saved me plants while I was traveling.",
        "My team defended my idea when I wasn't present at the meeting.",
        "The wedding guests accommodated last-minute changes with good humor."
    ]

    social_contexts_positive = [
        "Without being asked.",
        "Knowing I was struggling.",
        "With genuine kindness.",
        "Despite their busy schedules.",
        "Creating lasting memories.",
        "Strengthening our connection.",
        "Going out of their way.",
        "With thoughtful consideration.",
        "Making me feel included.",
        "Building community bonds."
    ]

    social_texts_negative = [
        "My friend shared my personal secret with the entire group.",
        "The host seated me at a table with strangers away from my friends.",
        "My roommate consistently leaves common areas messy despite agreements.",
        "The group chat suddenly went silent when I joined the conversation.",
        "My colleague took credit for my work during the team presentation.",
        "The neighbors complained about noise during our permitted daytime gathering.",
        "My relative made uncomfortable political comments throughout dinner.",
        "The carpool driver frequently arrives late without notification.",
        "My friend canceled our plans last minute for the third time this month.",
        "The committee excluded me from important decision-making discussions.",
        "The team member criticized my contribution in front of everyone.",
        "My classmates formed study groups but didn't include me.",
        "The family gathering turned into an unwelcome intervention about my career.",
        "My partner forgot our significant anniversary despite multiple reminders.",
        "The community board rejected my proposal without proper consideration."
    ]

    social_contexts_negative = [
        "Breaking my trust completely.",
        "Without any explanation.",
        "Knowing it would hurt me.",
        "With obvious indifference.",
        "Damaging our relationship.",
        "Creating unnecessary tension.",
        "In front of mutual friends.",
        "Despite previous promises.",
        "Showing clear disrespect.",
        "Making me feel excluded."
    ]

    social_texts_neutral = [
        "The neighborhood meeting covered standard agenda items in thirty minutes.",
        "My colleague mentioned potential schedule changes for next month.",
        "The group decided on rotating hosts for future gatherings.",
        "The committee meets every second Tuesday at the community center.",
        "My classmates discussed possible topics for the presentation assignment.",
        "The family exchanged regular updates through our shared calendar app.",
        "The roommates established a system for dividing household expenses.",
        "The organization announced elections for board positions next quarter.",
        "My friend group typically chooses restaurants that accommodate all diets.",
        "The team follows a round-robin approach for weekly progress reports.",
        "The club requires dues payment at the beginning of each semester.",
        "My department rotates responsibility for meeting minutes among members.",
        "The carpool arrangement alternates drivers based on a monthly schedule.",
        "The online forum moderates comments according to posted guidelines.",
        "The volunteer group communicates primarily through the official email list."
    ]

    social_contexts_neutral = [
        "As previously discussed.",
        "Following established patterns.",
        "According to group norms.",
        "Through regular channels.",
        "Per usual social protocol.",
        "As is customary.",
        "Without notable reaction.",
        "Maintaining normal dynamics.",
        "In our typical manner.",
        "Following social conventions."
    ]

    # =========== 5. Trải nghiệm giáo dục ===========
    education_texts_positive = [
        "The professor extended the deadline when I explained my circumstances.",
        "The online course offered practical skills I could immediately apply.",
        "My study group helped me understand complex concepts through examples.",
        "The workshop provided hands-on experience with professional equipment.",
        "The teacher created engaging activities that made learning enjoyable.",
        "The tutoring sessions significantly improved my test scores.",
        "The guest lecturer shared fascinating real-world applications of the theory.",
        "The curriculum challenged me to develop critical thinking skills.",
        "The feedback on my assignment highlighted both strengths and growth areas.",
        "The instructor adapted teaching methods to accommodate different learning styles.",
        "The scholarship committee recognized my academic achievements and potential.",
        "The educational software made practicing difficult concepts actually fun.",
        "The library staff went beyond expectations to help with my research.",
        "The international exchange program broadened my cultural understanding.",
        "The academic advisor helped map out a perfect course sequence for my goals."
    ]

    education_contexts_positive = [
        "Transforming my understanding completely.",
        "Building my confidence substantially.",
        "Opening new career possibilities.",
        "Exceeding course requirements.",
        "Making difficult concepts accessible.",
        "Inspiring deeper exploration.",
        "Encouraging intellectual curiosity.",
        "Through innovative approaches.",
        "With personalized attention.",
        "Creating lasting knowledge."
    ]

    education_texts_negative = [
        "The professor graded assignments inconsistently based on personal preference.",
        "The online platform crashed during our timed final examination.",
        "The required textbook contained numerous errors and outdated information.",
        "The instructor consistently arrived late, cutting our class time short.",
        "The group project partner contributed nothing but received equal credit.",
        "The academic advisor gave me incorrect information about graduation requirements.",
        "The expensive course covered only basic information available for free online.",
        "The teaching assistant was unreachable during posted office hours.",
        "The laboratory equipment was insufficient for all students to participate.",
        "The instructor responded defensively to questions about course material.",
        "The department canceled essential classes without offering alternatives.",
        "The practicum placement bore little resemblance to the description provided.",
        "The feedback on assignments was too vague to guide improvement.",
        "The lecture content rarely aligned with the stated learning objectives.",
        "The study materials provided inadequate preparation for the comprehensive exam."
    ]

    education_contexts_negative = [
        "Wasting valuable tuition money.",
        "Hindering academic progress.",
        "Creating unnecessary stress.",
        "Despite student complaints.",
        "Affecting many students negatively.",
        "Contrary to university policies.",
        "Undermining learning objectives.",
        "Without academic justification.",
        "Damaging educational outcomes.",
        "Ignoring accessibility needs."
    ]

    education_texts_neutral = [
        "The course requires three essays and one final examination for completion.",
        "The library maintains regular hours with extended times during exam periods.",
        "The department offers both introductory and advanced levels of the subject.",
        "The syllabus includes required readings from various academic sources.",
        "The lecture hall accommodates two hundred students with standard seating.",
        "The online platform uses standard authentication for student access.",
        "The laboratory sessions occur bi-weekly according to the schedule.",
        "The registration period opens based on accumulated credit hours.",
        "The textbook covers fundamental theories with supporting examples.",
        "The certification requires completing both written and practical assessments.",
        "The academic calendar includes standard breaks during traditional holidays.",
        "The campus shuttle operates between main buildings on thirty-minute intervals.",
        "The degree program consists of core requirements and elective options.",
        "The scholarship application requires academic transcripts and reference letters.",
        "The research database provides access to journal articles within the subscription."
    ]

    education_contexts_neutral = [
        "According to the syllabus.",
        "As stated in the handbook.",
        "Following standard procedure.",
        "Per academic requirements.",
        "Like previous course offerings.",
        "Based on curriculum guidelines.",
        "Within institutional norms.",
        "Following educational standards.",
        "As outlined in orientation.",
        "In line with department policy."
    ]

    # =========== 6. Công nghệ ===========
    tech_texts_positive = [
        "The software update fixed the security vulnerabilities and improved performance.",
        "The new algorithm reduced processing time by nearly sixty percent.",
        "My smart home system learns my preferences without requiring manual input.",
        "The company quickly patched the bug after users reported the issue.",
        "The cross-platform application works seamlessly across all my devices.",
        "The open-source community contributed innovative solutions to our project.",
        "The user interface redesign makes navigation intuitive and accessible.",
        "The cloud backup saved all my documents after my hard drive failed.",
        "The privacy features give me complete control over my personal data.",
        "The automated testing framework caught critical errors before deployment.",
        "The collaboration tools enabled real-time productivity across time zones.",
        "The customer support engineer resolved my technical issue in minutes.",
        "The accessibility features make the application usable for everyone.",
        "The documentation includes helpful examples for common use cases.",
        "The developer API allows endless customization for specific needs."
    ]

    tech_contexts_positive = [
        "Exceeding industry benchmarks.",
        "After the recent code overhaul.",
        "Without requiring advanced skills.",
        "Setting new standards.",
        "Through clever engineering.",
        "Improving existing workflows.",
        "With backward compatibility.",
        "Using minimal system resources.",
        "Through innovative design.",
        "Following user feedback."
    ]

    tech_texts_negative = [
        "The software constantly crashes when performing basic functions.",
        "The automatic update deleted my saved configurations without warning.",
        "The expensive subscription requires additional purchases for essential features.",
        "The privacy policy allows selling user data to third-party advertisers.",
        "The customer support transferred me between departments for two hours.",
        "The application interface changed completely without providing any tutorial.",
        "The cloud service lost important files during their server migration.",
        "The security breach exposed sensitive customer information for weeks.",
        "The constant notifications cannot be disabled despite settings changes.",
        "The supposed time-saving features actually require more steps than before.",
        "The battery drain issue persists despite multiple promised fixes.",
        "The compatibility problems prevent usage with standard industry tools.",
        "The automatic correction consistently makes inappropriate substitutions.",
        "The search function returns irrelevant results regardless of query specificity.",
        "The platform introduced advertisements in previously uninterrupted experiences."
    ]

    tech_contexts_negative = [
        "Despite multiple user complaints.",
        "After promising reliability.",
        "Following the expensive upgrade.",
        "Contradicting their marketing claims.",
        "Without any prior notification.",
        "Against industry security standards.",
        "Ignoring user feedback completely.",
        "Breaking core functionality.",
        "Disregarding accessibility needs.",
        "Requiring unnecessary workarounds."
    ]

    tech_texts_neutral = [
        "The system requires 8GB of RAM and 100GB of available storage space.",
        "The application updates automatically when connected to WiFi networks.",
        "The platform supports standard file formats including PDF and DOCX.",
        "The settings menu contains customization options for notifications.",
        "The subscription includes access to the current version plus updates.",
        "The login process requires two-factor authentication for security.",
        "The database performs regular backups according to the schedule.",
        "The interface displays information in the system's default language.",
        "The hardware connects through standard USB or Bluetooth protocols.",
        "The documentation describes both basic and advanced functionalities.",
        "The reporting feature exports data in CSV or JSON formats.",
        "The installation process takes approximately fifteen minutes to complete.",
        "The account management allows adding multiple authorized users.",
        "The compatibility extends to operating systems released within five years.",
        "The application uses standard keyboard shortcuts for common actions."
    ]

    tech_contexts_neutral = [
        "As stated in specifications.",
        "According to documentation.",
        "Following standard protocols.",
        "Per system requirements.",
        "Using default settings.",
        "In line with industry standards.",
        "As designed by developers.",
        "Under normal operating conditions.",
        "Based on current version.",
        "Within expected parameters."
    ]

    # =========== 7. Du lịch ===========
    travel_texts_positive = [
        "The local guide showed us hidden spots tourists rarely discover.",
        "The boutique hotel upgraded our room to a sea view without extra charge.",
        "The walking tour provided fascinating historical context about the city.",
        "The restaurant staff translated the entire menu and made perfect recommendations.",
        "The trail offered breathtaking views after the challenging climb.",
        "The museum curator gave us a private tour highlighting special exhibits.",
        "The transportation system made navigating the foreign city effortless.",
        "The homestay family included us in their traditional celebration.",
        "The boat tour allowed us to see marine wildlife in their natural habitat.",
        "The local festival embraced visitors with warm hospitality.",
        "The remote cabin provided perfect isolation with unexpected luxuries.",
        "The street food vendor prepared dishes according to our spice preferences.",
        "The historic site preservation maintained authenticity while ensuring accessibility.",
        "The sunset view from the rooftop terrace exceeded all expectations.",
        "The cultural workshop taught us traditional crafts with patient instruction."
    ]

    travel_contexts_positive = [
        "Making the trip unforgettable.",
        "Beyond what guidebooks mentioned.",
        "Creating authentic experiences.",
        "Highlighting local culture beautifully.",
        "Away from tourist crowds.",
        "With personal touches throughout.",
        "Showcasing hidden treasures.",
        "Through local connections.",
        "During perfect weather conditions.",
        "With unexpected opportunities."
    ]

    travel_texts_negative = [
        "The hotel room looked nothing like the professionally photographed website.",
        "The tour operator canceled without refund due to insufficient bookings.",
        "The crowded attraction allowed no opportunity to actually see anything.",
        "The restaurant served identical microwaved meals at premium 'authentic' prices.",
        "The beach was covered in trash despite its 'pristine paradise' marketing.",
        "The flight attendants treated questions as major impositions.",
        "The resort added numerous undisclosed fees at checkout.",
        "The famous landmark was completely covered in scaffolding during renovation.",
        "The expensive day trip spent most time at gift shops instead of attractions.",
        "The all-inclusive package excluded most activities worth experiencing.",
        "The cruise ship rooms suffered from constant mechanical noise.",
        "The popular hiking path was dangerously overcrowded with unprepared tourists.",
        "The airport shuttle service left without passengers despite confirmed bookings.",
        "The guided tour rushed through significant sites while lingering at souvenir stores.",
        "The luxury accommodation had visible mold and pest problems."
    ]

    travel_contexts_negative = [
        "Despite premium pricing.",
        "Ruining vacation plans completely.",
        "Contrary to advertised experiences.",
        "Wasting limited travel time.",
        "Without any prior notification.",
        "Ignoring safety standards.",
        "After explicit confirmation.",
        "With dismissive staff attitude.",
        "Misrepresenting actual conditions.",
        "During peak tourist season."
    ]

    travel_texts_neutral = [
        "The museum opens from 9AM to 5PM with last entry at 4:30PM.",
        "The hotel requires a credit card for incidental charges at check-in.",
        "The tour includes transportation but meals are purchased separately.",
        "The national park charges standard entrance fees for vehicles and pedestrians.",
        "The train departs hourly from the main station to downtown.",
        "The local currency is accepted at most established businesses.",
        "The boat tour operates weather permitting with safety guidelines.",
        "The historical site provides information in multiple languages.",
        "The resort offers both buffet and à la carte dining options.",
        "The hiking trail extends approximately five miles with moderate elevation.",
        "The airport shuttle runs according to a published schedule.",
        "The guided tour covers major landmarks over approximately three hours.",
        "The accommodations include standard amenities like WiFi and breakfast.",
        "The wildlife sanctuary maintains viewing areas at designated points.",
        "The marketplace operates during daylight hours with vendor variation."
    ]

    travel_contexts_neutral = [
        "According to their brochure.",
        "As listed on official website.",
        "Standard for this destination.",
        "Similar to comparable attractions.",
        "Following tourism regulations.",
        "As indicated on travel guides.",
        "Per visitor information.",
        "Typical for the region.",
        "Based on seasonal schedules.",
        "As mentioned during booking."
    ]

    # =========== 8. Trải nghiệm sức khỏe ===========
    health_texts_positive = [
        "The doctor took time to explain all treatment options in understandable terms.",
        "The new medication eliminated my symptoms without uncomfortable side effects.",
        "The physical therapist designed a program specifically addressing my needs.",
        "The fitness class modifications allowed me to participate despite limitations.",
        "The nutritionist created a sustainable meal plan fitting my lifestyle.",
        "The mental health hotline provided immediate support during my crisis.",
        "The hospital staff coordinated perfectly during my emergency situation.",
        "The wellness retreat gave me practical tools for managing daily stress.",
        "The specialist consulted colleagues to ensure the best treatment approach.",
        "The medical office scheduled my urgent concern on the same day.",
        "The surgery results exceeded expectations with minimal recovery time.",
        "The preventative screening detected an issue early, allowing simple treatment.",
        "The telehealth service provided expert consultation without travel requirements.",
        "The pharmacy staff identified a potential medication interaction before dispensing.",
        "The support group connected me with others sharing similar experiences."
    ]

    health_contexts_positive = [
        "When I needed it most.",
        "Making a difficult time easier.",
        "With compassionate approach.",
        "Transforming my wellbeing completely.",
        "After previous unsuccessful attempts.",
        "With patient-centered care.",
        "Despite complex circumstances.",
        "Restoring my confidence.",
        "Through evidence-based methods.",
        "Providing genuine relief."
    ]

    health_texts_negative = [
        "The doctor dismissed my symptoms without ordering appropriate tests.",
        "The hospital billed me for services never actually provided.",
        "The fitness instructor pushed dangerous movements despite my stated injury.",
        "The specialist kept me waiting two hours past my appointment time.",
        "The medical office lost my test results and required retesting.",
        "The pharmacy repeatedly filled my prescription incorrectly despite verification.",
        "The insurance denied coverage despite prior authorization approval.",
        "The wellness center oversold memberships creating impossible overcrowding.",
        "The surgeon appeared rushed and failed to address my pre-operative questions.",
        "The dietitian recommended products they personally sold at marked-up prices.",
        "The therapy session focused on insurance paperwork rather than treatment.",
        "The emergency room prioritized minor issues over serious conditions.",
        "The supplement caused adverse reactions not mentioned in warnings.",
        "The medical device malfunctioned consistently without company resolution.",
        "The healthcare portal exposed private information during their system update."
    ]

    health_contexts_negative = [
        "Despite my explicit symptoms.",
        "Causing unnecessary suffering.",
        "Without proper explanation.",
        "Ignoring established protocols.",
        "Creating additional problems.",
        "Against medical guidelines.",
        "With apparent indifference.",
        "Worsening my condition significantly.",
        "Despite multiple complaints.",
        "Breaking patient confidentiality."
    ]

    health_texts_neutral = [
        "The clinic operates during standard business hours with weekend urgent care.",
        "The prescription requires taking medication with food twice daily.",
        "The insurance coverage includes preventative care with standard copayments.",
        "The medical forms must be submitted prior to initial appointments.",
        "The facility provides both self-parking and valet options for patients.",
        "The laboratory results typically become available within three business days.",
        "The dietary guidelines recommend standard portions according to nutritional needs.",
        "The physical therapy exercises should be performed daily as demonstrated.",
        "The specialist referral requires documentation from your primary physician.",
        "The screening follows standard protocols established by health authorities.",
        "The patient portal allows access to personal medical records and appointments.",
        "The vaccination schedule follows recommendations for specific age groups.",
        "The pharmacy fills prescriptions during regular business hours with identification.",
        "The telehealth option requires basic technology with camera capabilities.",
        "The treatment plan includes follow-up appointments at specified intervals."
    ]

    health_contexts_neutral = [
        "According to medical protocol.",
        "Following standard procedures.",
        "As commonly prescribed.",
        "Per health guidelines.",
        "Based on current practices.",
        "Within normal parameters.",
        "As medically indicated.",
        "Under typical circumstances.",
        "Following established guidelines.",
        "As routinely recommended."
    ]

    # =========== Tổng hợp tất cả các danh mục ===========
    # Cấu trúc danh sách dữ liệu
    data = []
    categories = [
        (work_texts_positive, work_contexts_positive, "Positive"),
        (work_texts_negative, work_contexts_negative, "Negative"),
        (work_texts_neutral, work_contexts_neutral, "Neutral"),
        (product_texts_positive, product_contexts_positive, "Positive"),
        (product_texts_negative, product_contexts_negative, "Negative"),
        (product_texts_neutral, product_contexts_neutral, "Neutral"),
        (service_texts_positive, service_contexts_positive, "Positive"),
        (service_texts_negative, service_contexts_negative, "Negative"),
        (service_texts_neutral, service_contexts_neutral, "Neutral"),
        (social_texts_positive, social_contexts_positive, "Positive"),
        (social_texts_negative, social_contexts_negative, "Negative"),
        (social_texts_neutral, social_contexts_neutral, "Neutral"),
        (education_texts_positive, education_contexts_positive, "Positive"),
        (education_texts_negative, education_contexts_negative, "Negative"),
        (education_texts_neutral, education_contexts_neutral, "Neutral"),
        (tech_texts_positive, tech_contexts_positive, "Positive"),
        (tech_texts_negative, tech_contexts_negative, "Negative"),
        (tech_texts_neutral, tech_contexts_neutral, "Neutral"),
        (travel_texts_positive, travel_contexts_positive, "Positive"),
        (travel_texts_negative, travel_contexts_negative, "Negative"),
        (travel_texts_neutral, travel_contexts_neutral, "Neutral"),
        (health_texts_positive, health_contexts_positive, "Positive"),
        (health_texts_negative, health_contexts_negative, "Negative"),
        (health_texts_neutral, health_contexts_neutral, "Neutral")
    ]

    # Tạo dữ liệu cho từng danh mục
    for texts, contexts, label in categories:
        samples_per_category = num_samples // len(categories)
        for _ in range(samples_per_category):
            data.append({
                'text': random.choice(texts),
                'context': random.choice(contexts),
                'label': label
            })

    # Đảm bảo đủ số lượng mẫu
    while len(data) < num_samples:
        category = random.choice(categories)
        data.append({
            'text': random.choice(category[0]),
            'context': random.choice(category[1]),
            'label': category[2]
        })

    # Kiểm tra độ dài của văn bản và ngữ cảnh
    filtered_data = []
    for item in data:
        text_words = len(item['text'].split())
        context_words = len(item['context'].split())

        # Đảm bảo text < 50 từ và context < 20 từ
        if text_words <= 50 and context_words <= 20:
            filtered_data.append(item)

    # Đảm bảo có ít nhất 500 mẫu sau khi lọc
    final_data = filtered_data[:min(500, len(filtered_data))]

    # Trộn ngẫu nhiên dữ liệu
    random.shuffle(final_data)

    # Tạo DataFrame và lưu vào CSV
    df = pd.DataFrame(final_data)

    # Thêm thông tin về số lượng mẫu theo nhãn
    positive_count = sum(1 for item in final_data if item['label'] == 'Positive')
    negative_count = sum(1 for item in final_data if item['label'] == 'Negative')
    neutral_count = sum(1 for item in final_data if item['label'] == 'Neutral')

    print(f"Tổng số mẫu: {len(final_data)}")
    print(f"Số mẫu tích cực: {positive_count}")
    print(f"Số mẫu tiêu cực: {negative_count}")
    print(f"Số mẫu trung tính: {neutral_count}")

    # Tạo thư mục nếu chưa tồn tại
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)

    # Lưu vào file CSV
    df.to_csv(output_path, index=False)
    print(f"Dữ liệu đã được lưu vào {output_path}")

    return output_path

# If run directly, generate sample data
if __name__ == "__main__":
    generate_sample_data(num_samples=510)  # Generate slightly more than 500 samples