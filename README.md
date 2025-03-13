# LLM Eksperimentering

Dette er et verktøy for å eksperimentere med og sammenligne forskjellige språkmodeller (LLMs) via deres API-er.

## Funksjoner

- **Modellstøtte**:
  - OpenAI: GPT-3.5 Turbo, GPT-4o, GPT-4o mini
  - Anthropic: Claude 3.7 Sonnet, Claude 3.5 Sonnet, Claude 3.5 Haiku, Claude 3 Opus, Claude 3 Haiku
- **Sammenligningsmodus**: Sammenlign to modeller side ved side med samme prompt
- **Prisestimering**: Detaljert estimering av token-bruk og kostnader
- **API-statistikk**: Sporing av antall API-kall per modell
- **Visuelt forbedret grensesnitt**:
  - Emojier for modelltyper
  - Kompakt modellinformasjon
  - Delt samtalevisning i sammenligningsmodus
  - Hover-funksjonalitet for prisinfo
- **System prompt**: Definer systemprompter for å styre modellens oppførsel

## Installasjon

1. Installer avhengigheter:
```bash
pip install -r requirements.txt
```

2. Kopier `.env.example` til `.env`:
```bash
cp .env.example .env
```

3. Legg til dine API-nøkler i `.env` filen:
- Få en OpenAI API-nøkkel fra: https://platform.openai.com/api-keys
- Få en Anthropic API-nøkkel fra: https://console.anthropic.com/

## Kjør applikasjonen

```bash
streamlit run app.py
```

## Bruk

### Standard modus
1. Velg leverandør (OpenAI eller Anthropic)
2. Velg ønsket språkmodell fra nedtrekkslisten
3. Skriv inn din prompt i tekstfeltet
4. (Valgfritt) Tilpass system prompt
5. Klikk på "Send" for å generere svar

### Sammenligningsmodus
1. Aktiver "Sammenligningsmodus" med toggle-knappen
2. Velg to forskjellige modeller (kan være fra forskjellige leverandører)
3. Skriv inn din prompt
4. Klikk på "Send" for å generere svar fra begge modellene samtidig
5. Sammenlign svarene side ved side

### Prisestimering
- Se estimert token-bruk og kostnad i høyre sidepanel
- Hover over "💰" ikonet under modellnavnet for detaljert prisinformasjon
- Se API-kallstatistikk i ekspanderbaren i høyre sidepanel

## Sikkerhet

- Ikke del dine API-nøkler med andre
- `.env` filen er lagt til i `.gitignore` for å unngå at nøkler blir delt
- Hold dependencies oppdatert for beste sikkerhet 