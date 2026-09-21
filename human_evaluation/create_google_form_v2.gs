/**
 * Creates the human-evaluation Google Form automatically.
 *
 * How to run it:
 *   1. Go to script.google.com and choose "New project".
 *   2. Delete whatever is in the editor, paste this whole file, and press Save.
 *   3. Press Run. Google asks for permission the first time - allow it.
 *   4. The execution log prints the form's edit link and its public link.
 *
 * Nothing in the form names a model. Raters see only Response A, B and C.
 */

var ITEMS = [
  {
    "n": 1,
    "msg": "My children recently started school again and it made me feel a bit apprehensive. They are growing up so quick, and I know it won't be long before they sadly move out of the house :(",
    "replies": [
      {
        "label": "A",
        "text": "I understand that starting back at school can make parents like us feel anxious too. It's natural for us to miss our children when we don't see them regularly. But remember, they're growing up fast and moving on from home is part of becoming independent adults. Everything happens for a reason, even if we can't always see why right now. How do you think about this situation?"
      },
      {
        "label": "B",
        "text": "I can really understand how bittersweet and a bit scary that can be. It's such a big transition for both you and the kids. Knowing they're growing and learning is amazing, but it does bring a lot of emotions. How do you think they are adjusting to being back in school?"
      },
      {
        "label": "C",
        "text": "I understand how starting school again can bring a mix of excitement and apprehension, especially when thinking about your children growing up so fast and the bittersweet moments of them moving out. It's perfectly normal to feel a bit anxious about these changes. Remember, these are just stepping stones in their journey, and you are a constant source of love and support for them. How are you feeling about their upcoming school event?"
      }
    ]
  },
  {
    "n": 2,
    "msg": "I took a test for college the other day, it was basically our midterm.",
    "replies": [
      {
        "label": "A",
        "text": "I can really understand how nerve-wracking that must have been, especially since it feels like so much is riding on it. It's great that you're taking steps to prepare. How do you think it went?"
      },
      {
        "label": "B",
        "text": "I can sense your excitement and anticipation from taking your college test. It's a significant milestone, and your hard work is paying off. Remember, each step forward brings you closer to your dream college. How are you feeling right now?"
      },
      {
        "label": "C",
        "text": "Your test must have felt pretty challenging, huh? I'm here to listen if you want to talk about it. Is there anything specific that's on your mind right now?"
      }
    ]
  },
  {
    "n": 3,
    "msg": "Been having a good day, relaxing on my couch watching Netflix. I got a nice cup of hot chocolate and my cat curled up in my lap.",
    "replies": [
      {
        "label": "A",
        "text": "It sounds like you're really enjoying some peaceful moments to unwind. It's wonderful to have those small, cozy pleasures that can make a big difference. You deserve this downtime to recharge. How long do you usually spend watching Netflix before you decide to do something else?"
      },
      {
        "label": "B",
        "text": "It sounds like you're having a wonderful, relaxing day. The comfort of your hot chocolate, Netflix, and your cat's warmth in your lap really seems to be putting you at ease. Remember, moments like these are perfect for simply enjoying the peace around you. How are you feeling right now?"
      },
      {
        "label": "C",
        "text": "That sounds incredibly peaceful! It's wonderful that you're finding moments like this to relax and enjoy yourself. Your well-deserved break feels so much better than anything else."
      }
    ]
  },
  {
    "n": 4,
    "msg": "My Calculus final was no joke! I started studying hard two weeks before the test.",
    "replies": [
      {
        "label": "A",
        "text": "Your dedication to studying Calculus really paid off, and it's fantastic to see how prepared you felt for your final. Your hard work and commitment truly deserve recognition. Remember, your knowledge and understanding of the subject matter are powerful tools. Is there anything else on your mind after this accomplishment?"
      },
      {
        "label": "B",
        "text": "I'm so glad to hear that you've been putting in the effort! That's truly admirable. You're doing great by taking control of your studies. What next steps do you plan to take after this exam?"
      },
      {
        "label": "C",
        "text": "I can only imagine how stressful those last few days must have been for you. It sounds like you put in a lot of effort to prepare, but it can still be overwhelming. You did everything you could to be ready. How do you think you did on the test?"
      }
    ]
  },
  {
    "n": 5,
    "msg": "So I moved away from my parents not too long ago. It's been a big change.",
    "replies": [
      {
        "label": "A",
        "text": "I hear how moving away from your parents must feel isolating and challenging. It's natural to miss their presence and feel the weight of this new distance. Remember, even though you're physically apart, the love and connection you share can transcend distance. How have you been feeling emotionally since the move?"
      },
      {
        "label": "B",
        "text": "I'm really sorry to hear about that move. Moving away can be tough, but it's also an exciting step into new opportunities. How do you find yourself adjusting to this change?"
      },
      {
        "label": "C",
        "text": "I can only imagine how overwhelming and bittersweet that must be for you. Leaving home can be so hard, but it's also a brave step towards independence. You're doing great by taking this step. How are you finding life on your own so far?"
      }
    ]
  },
  {
    "n": 6,
    "msg": "If my upstairs neighbor wakes me at 3am again I think I'll explode! I get that she has to go to work, but why not get organized the night before for goodness sake?!",
    "replies": [
      {
        "label": "A",
        "text": "I hear how frustrating it is to be woken up by your upstairs neighbor at 3am. It's perfectly understandable to feel annoyed when something that should be predictable disrupts your sleep. Maybe try talking to her about setting up a more predictable routine? How do you feel about discussing this with her?"
      },
      {
        "label": "B",
        "text": "I understand how frustrating it must be for you when your upstairs neighbor disturbs your sleep. It's understandable that this could be stressful. Remember, waking up early can be challenging even under normal circumstances. What specifically do you need from her so that you can manage this better?"
      },
      {
        "label": "C",
        "text": "I can really understand how frustrating and overwhelming it must be to have your sleep disrupted like that. It sounds like you're dealing with a lot of stress and maybe even some anger about the situation. Knowing that she has responsibilities is important, but it's understandable to feel like things could be handled better. Does talking to her about it seem like a good idea to you?"
      }
    ]
  },
  {
    "n": 7,
    "msg": "In college I was in a class with an older woman who stole my ideas for a paper.",
    "replies": [
      {
        "label": "A",
        "text": "I'm so sorry that happened to you. It's hard when someone takes credit for work you put in. Just remember, you can always reach out for support from friends or professionals like counselors or mentors. The most important thing is that you stand strong and don't let this setback define you. What do you think about sharing more details about how you're handling it right now?"
      },
      {
        "label": "B",
        "text": "I can only imagine how hurt and frustrated you must feel. It's so unfair when others take credit for our hard work. You deserve recognition for your efforts. Did this experience change how you approach sharing your ideas in the future?"
      },
      {
        "label": "C",
        "text": "I hear you, and it's completely valid to feel annoyed in this situation. It's tough when someone in a position of authority takes credit for your work. Remember, your ideas and contributions are valuable and deserve recognition. Just know that this doesn't define you or your abilities. Keep focusing on your passion and dedication. Would you like to talk more about this?"
      }
    ]
  },
  {
    "n": 8,
    "msg": "My dog disappeared while I was away at college.",
    "replies": [
      {
        "label": "A",
        "text": "I'm so sorry about that. Losing a pet can be incredibly heartbreaking. Please know that there is no time limit on grief; it's okay to grieve for days or even months after losing someone. How are you finding comfort during this difficult time?"
      },
      {
        "label": "B",
        "text": "I'm so sorry to hear that your dog went missing; it must be incredibly worrying and heartbreakingly sad for you. Losing a pet can feel like a huge loss, but know that you're not alone in this pain. Have you had any updates since you found out?"
      },
      {
        "label": "C",
        "text": "I hear how devastating this must be for you. It's completely understandable to feel overwhelmed and heartbroken when someone so close to you vanishes. It's okay to grieve the absence of your beloved pet. Remember, their spirit and memories will always bring comfort and love to your heart. How are you holding up?"
      }
    ]
  }
];

var QUESTIONS = [
  {
    "title": "How well does this reply recognise and validate the emotions expressed in the message?",
    "help": "1 = Ignores or dismisses the feelings   3 = Acknowledges them somewhat generically   5 = Clearly understands and validates the feelings"
  },
  {
    "title": "How well does this reply address the person's specific situation rather than giving a generic response?",
    "help": "1 = Generic, could apply to anyone   3 = Partially specific   5 = Clearly tailored to what was shared"
  },
  {
    "title": "If you had written this message, how supported or comforted would you feel by this reply?",
    "help": "1 = Not at all / worse   3 = Neutral   5 = Very supported"
  }
];

var PREFERENCE_Q = "Overall, which reply responds best to this person's feelings?";

var CONSENT = "Thank you for helping with this study.\n\nYou will read 8 short messages from people describing something they are going through. For each message you will see three possible replies, and you will rate each reply on three simple 1-5 scales. There are no right or wrong answers - we want your honest impression.\n\nThis takes about 15 minutes. Your answers are completely anonymous: we do not collect your name or email address. Some messages describe sad or stressful situations, and you may stop at any time.\n\nThe replies were written by different computer programs, but we do not say which is which, because that would change how you read them.\n\nBy continuing, you agree to take part.";

function createForm() {
  var form = FormApp.create('How do people respond to someone who is upset? (15 minutes)');
  form.setDescription(CONSENT);
  form.setProgressBar(true);
  try { form.setCollectEmail(false); } catch (e) { /* newer accounts already collect no email */ }
  form.setShuffleQuestions(false);

  form.addMultipleChoiceItem()
      .setTitle('I am 18 or older and I agree to take part in this study.')
      .setChoiceValues(['Yes', 'No'])
      .setRequired(true);

  for (var i = 0; i < ITEMS.length; i++) {
    var item = ITEMS[i];
    form.addPageBreakItem().setTitle('Situation ' + item.n + ' of ' + ITEMS.length);

    var labels = [];
    form.addSectionHeaderItem()
        .setTitle('Someone wrote:')
        .setHelpText('"' + item.msg + '"');
    for (var r = 0; r < item.replies.length; r++) {
      form.addSectionHeaderItem()
          .setTitle('Response ' + item.replies[r].label)
          .setHelpText(item.replies[r].text);
      labels.push('Response ' + item.replies[r].label);
    }

    for (var q = 0; q < QUESTIONS.length; q++) {
      form.addGridItem()
          .setTitle(QUESTIONS[q].title)
          .setHelpText(QUESTIONS[q].help)
          .setRows(labels)
          .setColumns(['1', '2', '3', '4', '5'])
          .setRequired(true);
    }

    form.addMultipleChoiceItem()
        .setTitle(PREFERENCE_Q)
        .setChoiceValues(labels)
        .setRequired(true);
  }

  form.addPageBreakItem().setTitle('Thank you');
  form.addParagraphTextItem()
      .setTitle('Anything you would like to add? (optional)')
      .setRequired(false);

  Logger.log('Edit the form here: ' + form.getEditUrl());
  Logger.log('Send this link to raters: ' + form.getPublishedUrl());
}
