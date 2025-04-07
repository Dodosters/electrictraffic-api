
# Business tariffs data
business_tariffs = [
  {
    "id": 1,
    "region": "Ростов-на-Дону",
    "provider": "ТНС энерго Ростов-на-Дону",
    "tariffTypes": [
      {
        "id": 1,
        "name": "Одноставочный",
        "rate": 7.2,
        "unit": "руб/кВтч"
      },
      {
        "id": 2,
        "name": "Двухставочный",
        "rates": [
          { "id": 1, "name": "День", "rate": 8.3, "unit": "руб/кВтч" },
          { "id": 2, "name": "Ночь", "rate": 3.1, "unit": "руб/кВтч" }
        ]
      },
      {
        "id": 3,
        "name": "Трехставочный",
        "rates": [
          { "id": 1, "name": "Пик", "rate": 9.5, "unit": "руб/кВтч" },
          { "id": 2, "name": "Полупик", "rate": 7.8, "unit": "руб/кВтч" },
          { "id": 3, "name": "Ночь", "rate": 3.1, "unit": "руб/кВтч" }
        ]
      }
    ]
  },
  {
    "id": 2,
    "region": "Таганрог",
    "provider": "ТНС энерго Ростов-на-Дону",
    "tariffTypes": [
      {
        "id": 1,
        "name": "Одноставочный",
        "rate": 7.1,
        "unit": "руб/кВтч"
      },
      {
        "id": 2,
        "name": "Двухставочный",
        "rates": [
          { "id": 1, "name": "День", "rate": 8.2, "unit": "руб/кВтч" },
          { "id": 2, "name": "Ночь", "rate": 3.0, "unit": "руб/кВтч" }
        ]
      },
      {
        "id": 3,
        "name": "Трехставочный",
        "rates": [
          { "id": 1, "name": "Пик", "rate": 9.3, "unit": "руб/кВтч" },
          { "id": 2, "name": "Полупик", "rate": 7.7, "unit": "руб/кВтч" },
          { "id": 3, "name": "Ночь", "rate": 3.0, "unit": "руб/кВтч" }
        ]
      }
    ]
  },
  {
    "id": 3,
    "region": "Шахты",
    "provider": "ТНС энерго Ростов-на-Дону",
    "tariffTypes": [
      {
        "id": 1,
        "name": "Одноставочный",
        "rate": 6.9,
        "unit": "руб/кВтч"
      },
      {
        "id": 2,
        "name": "Двухставочный",
        "rates": [
          { "id": 1, "name": "День", "rate": 8.0, "unit": "руб/кВтч" },
          { "id": 2, "name": "Ночь", "rate": 2.9, "unit": "руб/кВтч" }
        ]
      }
    ]
  },
  {
    "id": 4,
    "region": "Новочеркасск",
    "provider": "ТНС энерго Ростов-на-Дону",
    "tariffTypes": [
      {
        "id": 1,
        "name": "Одноставочный",
        "rate": 7.0,
        "unit": "руб/кВтч"
      },
      {
        "id": 2,
        "name": "Двухставочный",
        "rates": [
          { "id": 1, "name": "День", "rate": 8.1, "unit": "руб/кВтч" },
          { "id": 2, "name": "Ночь", "rate": 3.0, "unit": "руб/кВтч" }
        ]
      },
      {
        "id": 3,
        "name": "Трехставочный",
        "rates": [
          { "id": 1, "name": "Пик", "rate": 9.2, "unit": "руб/кВтч" },
          { "id": 2, "name": "Полупик", "rate": 7.6, "unit": "руб/кВтч" },
          { "id": 3, "name": "Ночь", "rate": 3.0, "unit": "руб/кВтч" }
        ]
      }
    ]
  },
  {
    "id": 5,
    "region": "Волгодонск",
    "provider": "ТНС энерго Ростов-на-Дону",
    "tariffTypes": [
      {
        "id": 1,
        "name": "Одноставочный",
        "rate": 6.8,
        "unit": "руб/кВтч"
      },
      {
        "id": 2,
        "name": "Двухставочный",
        "rates": [
          { "id": 1, "name": "День", "rate": 7.9, "unit": "руб/кВтч" },
          { "id": 2, "name": "Ночь", "rate": 2.8, "unit": "руб/кВтч" }
        ]
      }
    ]
  },
  {
    "id": 6,
    "region": "Батайск",
    "provider": "ТНС энерго Ростов-на-Дону",
    "tariffTypes": [
      {
        "id": 1,
        "name": "Одноставочный",
        "rate": 7.0,
        "unit": "руб/кВтч"
      },
      {
        "id": 2,
        "name": "Двухставочный",
        "rates": [
          { "id": 1, "name": "День", "rate": 8.1, "unit": "руб/кВтч" },
          { "id": 2, "name": "Ночь", "rate": 3.0, "unit": "руб/кВтч" }
        ]
      }
    ]
  }
]

# Personal tariffs data
personal_tariffs = [
  {
    "id": 1,
    "region": "Ростов-на-Дону",
    "provider": "ТНС энерго Ростов-на-Дону",
    "tariffTypes": [
      {
        "id": 1,
        "name": "Одноставочный",
        "rate": 5.60,
        "unit": "руб/кВтч"
      },
      {
        "id": 2,
        "name": "Двухзонный",
        "rates": [
          { "id": 1, "name": "День", "rate": 6.44, "unit": "руб/кВтч" },
          { "id": 2, "name": "Ночь", "rate": 2.24, "unit": "руб/кВтч" }
        ]
      },
      {
        "id": 3,
        "name": "Трехзонный",
        "rates": [
          { "id": 1, "name": "Пик", "rate": 7.28, "unit": "руб/кВтч" },
          { "id": 2, "name": "Полупик", "rate": 5.60, "unit": "руб/кВтч" },
          { "id": 3, "name": "Ночь", "rate": 2.24, "unit": "руб/кВтч" }
        ]
      }
    ]
  },
  {
    "id": 2,
    "region": "Таганрог",
    "provider": "ТНС энерго Ростов-на-Дону",
    "tariffTypes": [
      {
        "id": 1,
        "name": "Одноставочный",
        "rate": 5.50,
        "unit": "руб/кВтч"
      },
      {
        "id": 2,
        "name": "Двухзонный",
        "rates": [
          { "id": 1, "name": "День", "rate": 6.33, "unit": "руб/кВтч" },
          { "id": 2, "name": "Ночь", "rate": 2.20, "unit": "руб/кВтч" }
        ]
      }
    ]
  },
  {
    "id": 3,
    "region": "Шахты",
    "provider": "ТНС энерго Ростов-на-Дону",
    "tariffTypes": [
      {
        "id": 1,
        "name": "Одноставочный",
        "rate": 5.40,
        "unit": "руб/кВтч"
      },
      {
        "id": 2,
        "name": "Двухзонный",
        "rates": [
          { "id": 1, "name": "День", "rate": 6.21, "unit": "руб/кВтч" },
          { "id": 2, "name": "Ночь", "rate": 2.16, "unit": "руб/кВтч" }
        ]
      }
    ]
  },
  {
    "id": 4,
    "region": "Новочеркасск",
    "provider": "ТНС энерго Ростов-на-Дону",
    "tariffTypes": [
      {
        "id": 1,
        "name": "Одноставочный",
        "rate": 5.45,
        "unit": "руб/кВтч"
      },
      {
        "id": 2,
        "name": "Двухзонный",
        "rates": [
          { "id": 1, "name": "День", "rate": 6.27, "unit": "руб/кВтч" },
          { "id": 2, "name": "Ночь", "rate": 2.18, "unit": "руб/кВтч" }
        ]
      }
    ]
  }
]

# Energy providers data
providers = [
  {
    "id": 1,
    "name": "ТНС энерго Ростов-на-Дону",
    "regions": ["Ростов-на-Дону", "Таганрог", "Шахты", "Новочеркасск", "Волгодонск", "Батайск"],
    "website": "https://rostov.tns-e.ru/",
    "contacts": {
      "phone": "8 (863) 307-73-03",
      "email": "info@rostov.tns-e.ru"
    }
  }
]

# Analytics data
analytics_data = {
  "regionalComparison": [
    { "region": "Ростов-на-Дону", "averageRate": 7.2, "change": 5.0 },
    { "region": "Таганрог", "averageRate": 7.1, "change": 4.8 },
    { "region": "Шахты", "averageRate": 6.9, "change": 4.5 },
    { "region": "Новочеркасск", "averageRate": 7.0, "change": 4.7 },
    { "region": "Волгодонск", "averageRate": 6.8, "change": 4.3 },
    { "region": "Батайск", "averageRate": 7.0, "change": 4.7 },
    { "region": "Азов", "averageRate": 6.9, "change": 4.4 },
    { "region": "Каменск-Шахтинский", "averageRate": 6.7, "change": 4.2 }
  ],
  "yearlyTrends": [
    { "year": 2018, "averageRate": 5.1 },
    { "year": 2019, "averageRate": 5.5 },
    { "year": 2020, "averageRate": 6.0 },
    { "year": 2021, "averageRate": 6.3 },
    { "year": 2022, "averageRate": 6.7 },
    { "year": 2023, "averageRate": 7.0 },
    { "year": 2024, "averageRate": 7.2 },
    { "year": 2025, "averageRate": 7.5 }
  ],
  "forecastData": [
    { "year": 2025, "quarter": 2, "predictedRate": 7.6 },
    { "year": 2025, "quarter": 3, "predictedRate": 7.8 },
    { "year": 2025, "quarter": 4, "predictedRate": 7.9 },
    { "year": 2026, "quarter": 1, "predictedRate": 8.1 },
    { "year": 2026, "quarter": 2, "predictedRate": 8.3 }
  ]
}

# FAQ data
faq_data = [
  {
    "id": 1,
    "question": "Как рассчитать стоимость электроэнергии?",
    "answer": "Стоимость электроэнергии рассчитывается путем умножения объема потребленной электроэнергии на тариф. Объем потребления определяется как разница показаний счетчика на конец и начало расчетного периода."
  },
  {
    "id": 2,
    "question": "Какие бывают тарифы на электроэнергию в Ростовской области?",
    "answer": "В Ростовской области существуют следующие основные виды тарифов: одноставочный (единый тариф на всё время суток), двухзонный (день/ночь) и трехзонный (пик/полупик/ночь). Выбор тарифа зависит от характера потребления электроэнергии."
  },
  {
    "id": 3,
    "question": "Как часто меняются тарифы на электроэнергию в Ростовской области?",
    "answer": "Тарифы на электроэнергию в Ростовской области обычно пересматриваются раз в год. Изменение тарифов происходит 1 июля."
  },
  {
    "id": 4,
    "question": "Чем отличаются тарифы для юридических и физических лиц в Ростовской области?",
    "answer": "Тарифы для юридических лиц в Ростовской области обычно выше, чем для физических, и могут включать дополнительные составляющие, такие как плата за мощность. Кроме того, для юридических лиц могут применяться разные тарифы в зависимости от категории напряжения и других параметров."
  },
  {
    "id": 5,
    "question": "Как перейти на другой тариф в Ростовской области?",
    "answer": "Для перехода на другой тариф необходимо обратиться в офис \"ТНС энерго Ростов-на-Дону\" с соответствующим заявлением. В некоторых случаях может потребоваться замена счетчика, если существующий не поддерживает необходимый режим учета электроэнергии."
  }
]

# News data
news_data = [
  {
    "id": 1,
    "title": "Тарифы на электроэнергию в Ростовской области вырастут с 1 июля 2025 года",
    "date": "2025-04-01",
    "summary": "Региональная служба по тарифам Ростовской области утвердила повышение тарифов на электроэнергию с 1 июля 2025 года в среднем на 4.7%.",
    "content": "Региональная служба по тарифам Ростовской области утвердила повышение тарифов на электроэнергию с 1 июля 2025 года. Согласно постановлению, рост тарифов составит в среднем 4.7% по сравнению с действующими ставками. При этом в разных городах области повышение может отличаться в зависимости от местных условий.\n\nВ Ростове-на-Дону ожидается увеличение тарифов на 5.0%, в Таганроге — на 4.8%. Наименьший рост планируется в городе Каменск-Шахтинский — всего 4.2%.\n\nПредставители РСТ отмечают, что повышение тарифов ниже прогнозируемого уровня инфляции и направлено на поддержание инвестиционной активности в электроэнергетике Ростовской области."
  },
  {
    "id": 2,
    "title": "Новые правила расчета платы за электроэнергию для бизнеса в Ростовской области",
    "date": "2025-03-25",
    "summary": "Министерство промышленности и энергетики Ростовской области представило новую методику расчета стоимости электроэнергии для предприятий.",
    "content": "Министерство промышленности и энергетики Ростовской области разработало и представило новую методику расчета стоимости электроэнергии для предприятий. Согласно документу, с 1 сентября 2025 года вводится новый порядок учета мощности и потребления электроэнергии для юридических лиц.\n\nОсновные изменения касаются предприятий, работающих в часы пиковой нагрузки. Для них вводится дополнительный коэффициент, который будет стимулировать более равномерное потребление электроэнергии в течение суток.\n\nЭксперты отрасли отмечают, что новые правила потенциально могут снизить затраты на электроэнергию для компаний Ростовской области, которые смогут перенести часть производственных процессов на ночное время."
  },
  {
    "id": 3,
    "title": "Запущена программа энергоэффективности для предприятий Ростовской области",
    "date": "2025-03-15",
    "summary": "Правительство Ростовской области запустило новую программу повышения энергоэффективности для предприятий и населения.",
    "content": "Правительство Ростовской области запустило новую программу повышения энергоэффективности, рассчитанную на период 2025-2030 годов. Программа предусматривает комплекс мер по снижению энергопотребления как для предприятий, так и для населения.\n\nВ рамках программы предусмотрены субсидии на установку энергосберегающего оборудования, льготные кредиты на модернизацию энергосистем предприятий, а также информационная поддержка населения по вопросам экономии электроэнергии.\n\nПо оценкам Министерства промышленности и энергетики Ростовской области, реализация программы позволит снизить энергопотребление в регионе на 6-8% к 2030 году, что приведет к существенной экономии как для потребителей, так и для всей энергосистемы области."
  }
]
