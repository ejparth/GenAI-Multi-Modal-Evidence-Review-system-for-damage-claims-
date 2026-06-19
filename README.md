# Multi-Modal Damage Claim Verification System

A production-ready AI-powered system for reviewing damage claims using images, user conversations, claim history, and evidence requirements.

The system evaluates submitted evidence and determines whether a claim is:

- **Supported**
- **Contradicted**
- **Not Enough Information**

while providing explainable reasoning, risk assessment, and evidence validation.

---

## Overview

Damage claims often contain a combination of:

- User-provided images
- Short conversational descriptions
- Historical claim activity
- Evidence requirements

This project combines vision-language models, structured reasoning, and rule-based validation to assess whether the available evidence supports a reported issue.

The solution focuses on:

- Multi-image reasoning
- Evidence sufficiency assessment
- Risk detection
- Explainable decisions
- Cost-efficient processing
- Structured outputs

---

## Features

### Claim Understanding

Extracts key information from conversational claims:

- Issue type
- Affected object part
- Claimed damage description
- Confidence score

### Visual Evidence Analysis

Analyzes one or more submitted images to identify:

- Visible damage
- Object parts
- Damage severity
- Image quality
- Coverage completeness

### Evidence Validation

Determines whether submitted images satisfy minimum evidence requirements.

Examples:

- Required object part visible
- Damage clearly observable
- Sufficient image coverage

### Risk Assessment

Detects potential issues such as:

- Blurry images
- Cropped or obstructed views
- Wrong object
- Wrong object part
- Low-light conditions
- Missing damage visibility
- Suspicious claim history

### Explainable Decisions

Every prediction includes:

- Decision status
- Supporting image IDs
- Evidence reasoning
- Risk flags
- Severity estimate

---

## Supported Objects

### Vehicles

Supported damage categories include:

- Dent
- Scratch
- Crack
- Glass damage
- Missing parts
- Broken parts

### Laptops

Supported damage categories include:

- Cracked screens
- Broken hinges
- Keyboard damage
- Port damage
- Body damage

### Packages

Supported damage categories include:

- Torn packaging
- Crushed packaging
- Water damage
- Missing contents
- Label damage

---

## System Architecture

```text
User Claim
     │
     ▼
Claim Extraction Agent
     │
     ▼
Image Analysis Agent
     │
     ▼
Multi-Image Aggregator
     │
     ▼
Evidence Requirement Checker
     │
     ▼
Risk Assessment Agent
     │
     ▼
Decision Engine
     │
     ▼
Suctured Output


---
```
## Quick Start

### Clone the Repository

If you'd like to explore, test, or extend the project, clone the repository and run it locally.

```bash
git clone https://github.com/ejparth/GenAI-Multi-Modal-Evidence-Review-system-for-damage-claims-.git
cd GenAI-Multi-Modal-Evidence-Review-system-for-damage-claims-
