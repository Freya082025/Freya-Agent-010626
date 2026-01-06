Here is the comprehensive technical specification for the FDA 510(k) Agentic Review WOW Studio, based on the implemented architecture and design patterns.
Technical Specification: FDA 510(k) Agentic Review WOW Studio
Version: 1.0
Date: October 26, 2023
Status: Approved for Implementation
1. Executive Summary
1.1 Purpose
The FDA 510(k) Agentic Review WOW Studio is a specialized, web-based agentic AI application designed to accelerate and standardize the regulatory review process for medical devices. It assists regulatory professionals, data analysts, and product teams by employing a multi-agent system driven by Large Language Models (LLMs) to ingest, analyze, critique, and summarize FDA 510(k) submission materials.
1.2 Scope
This specification defines the frontend architecture, state management strategies, AI integration patterns, and user interface designs required to deliver a responsive, secure, and visually immersive experience ("WOW UI"). The system operates as a Single Page Application (SPA) leveraging the Google Gemini API for intelligence, with a focus on privacy (client-side data handling) and aesthetic flexibility (Painter Styles).
1.3 Key Objectives
Agentic Workflow: Enable users to select specialized AI agents (e.g., Triage, Labeling, Risk) to process submission documents.
Data Intelligence: Provide a "Data Lab" for conversational analysis of structured FDA datasets (CSV/JSON).
Productivity: Integrate an AI-enhanced note-taking environment for summarization, translation, and action extraction.
Visual Immersion: Implement a dynamic theming engine allowing users to switch environments based on famous painter styles.
2. System Architecture
2.1 Conceptual Architecture
The application follows a Client-Side Agentic Architecture. The browser acts as the orchestration layer, managing document context, user inputs, and agent configurations. It communicates directly with the Google Gemini API for inference tasks. There is no intermediate backend server for business logic, ensuring that sensitive submission data resides in the user's browser memory during the session (ephemeral state) unless explicitly exported.
2.2 Technology Stack
Core Framework: React 19 with TypeScript.
Build Tooling: ES Modules (via importmap for rapid prototyping/deployment without heavy bundling).
Styling Engine: Tailwind CSS with dynamic class injection for theming.
AI Integration: @google/genai SDK (Gemini 2.5 and 3.0 series models).
Visualization: recharts for data visualization.
Icons: lucide-react.
2.3 Data Flow
Input: User uploads files (Text/Markdown/CSV) or inputs API keys.
State: Data is normalized and stored in a central React AppState object.
Processing:
Agents: The AgentRunner constructs a prompt combining system instructions, document context, and user queries, sending it to Gemini.
Data Lab: The FdaDataLab aggregates dataset samples into a context window and manages a chat session.
Output: Structured Markdown, JSON metadata (tokens/latency), or visualization data is returned and rendered.
3. Frontend Application Design
3.1 Component Hierarchy
The application is structured as a tree of functional components:
App (Root Container, Layout Manager, Theme Injector)
Sidebar (Navigation, Global Settings, API Key, Jackpot/Theme Selection)
Main Content Area (Router/Tab Switcher)
AgentRunner (Document Ingestion, Agent Selection, Execution)
Dashboard (Metrics, Charts, Run Logs)
FdaDataLab (Dataset Management, Chat Interface)
AiNoteKeeper (Editor, Magic Actions Toolbar)
3.2 State Management
The application uses a centralized, monolithic state pattern passed down via props or context (React useState).
AppState Interface:
code
TypeScript
interface AppState {
  language: 'en' | 'zh-tw';
  theme: 'light' | 'dark';
  currentStyle: PainterStyle; // Active visual theme
  apiKey: string;             // User-provided API key
  documents: Document[];      // Uploaded submission files
  runLog: RunLogEntry[];      // History of agent executions
  fdaDatasets: FdaDataset[];  // Uploaded structured data
  fdaChatHistory: ChatMessage[]; // Data Lab conversation history
  notes: string;              // Current scratchpad content
}
This approach ensures strict synchronization between tabs (e.g., an agent run in the Runner immediately updates the Dashboard stats).
3.3 Theming Engine: "Painter Styles"
A unique feature is the "WOW UI" theming engine. Instead of standard color palettes, the app uses PainterStyle objects.
Mechanism:
Definition: Styles (Van Gogh, Monet, Dali, etc.) are defined in constants.ts.
Properties:
gradient: A complex Tailwind background gradient class string.
accent: Text colors for headers and active states.
cardBg: Backdrop blur and opacity settings (glassmorphism) for containers.
Application: The App component applies the gradient to the body. Child components (Sidebar, Dashboard) consume currentStyle.cardBg and currentStyle.accent to render their specific elements.
Jackpot Mode: A specialized logic in Sidebar cycles through styles rapidly (setInterval) before settling on a random choice, adding gamification.
4. Functional Modules
4.1 Authentication & Configuration (Sidebar)
API Key Handling:
Priority: process.env.API_KEY > User Input State.
Security: Input fields are masked (type="password"). Keys are never logged or stored in localStorage to prevent persistence risks on shared machines.
Internationalization (i18n):
A dictionary object TRANSLATIONS holds string resources keyed by en and zh-tw.
Sidebar toggles the state.language, triggering immediate re-renders of all text nodes.
4.2 Agent Runner Engine
This is the core workflow engine.
Agent Configuration:
Agents are defined in AGENTS constant array.
Each agent contains a system_prompt tailored for specific regulatory tasks (e.g., "Predicate Device Mapper", "Risk-Benefit Summarizer").
Defaults include model (e.g., gemini-2.5-flash) and temperature (0.2).
Context Construction:
When "Run Agent" is clicked, the engine concatenates all uploaded state.documents into a single context block: CONTEXT DOCUMENTS: \n --- DOCUMENT: {name} --- \n {content}.
User instructions are appended.
Execution:
Calls generateContent service.
On success, creates a RunLogEntry containing tokens used, duration, and output text.
Output is displayed in a read-only viewer with a "Use as Input" feedback loop capability.
4.3 Analytics Dashboard
Visualizes the efficiency and usage of the agent system.
Metrics:
Total Runs, Total Tokens, Unique Agents utilized.
Visualizations (via Recharts):
Bar Chart: "Runs by Agent" (Vertical layout).
Line Chart: "Token Usage Timeline" (Tokens vs. Run Sequence).
Log Table:
Displays recent runs with status badges (Success/Error), timestamps, and latency.
4.4 FDA Data Lab (RAG-Lite)
A module for interacting with structured data without a traditional database.
Ingestion:
Parses CSV/JSON/Text files into FdaDataset objects.
Calculates metadata (row counts) for quick preview.
Context Window Management:
To fit within LLM context windows, the app does not send entire datasets.
Strategy: It constructs a "meta-context" containing the dataset Name, Category, and a sample (first 500-1000 characters or ~20 rows).
Prompt Engineering: "You are an FDA Data Analyst... If data is missing, say so."
Chat Interface:
Maintains a linear conversation history (fdaChatHistory).
Uses chatWithData service which leverages ai.chats.create to maintain conversational context (turn-taking).
4.5 AI Note Keeper
An intelligent scratchpad for regulatory drafting.
Editor: Simple textarea allowing raw text entry.
"AI Magic" Operations:
Pre-defined prompt templates: Format, Summary, Translate, Actions, Expand.
Workflow:
User selects operation.
System wraps current note content with specific instruction (e.g., "Format the following notes into clean Markdown...").
LLM processes and replaces/appends content.
UI: Overlay loading state with "Processing..." spinner.
5. Data Models & Schemas
5.1 Agent Configuration
code
TypeScript
interface AgentConfig {
  id: string;            // Unique identifier (e.g., "intake_triage")
  name: string;          // Display name
  skill_number: number;  // Ordering index
  category: string;      // Grouping (e.g., "Labeling")
  default_model: string; // e.g., "gemini-2.5-flash"
  description: string;   // Tooltip text
  system_prompt: string; // The core personality/instruction
}
5.2 Run Log Entry
code
TypeScript
interface RunLogEntry {
  id: string;
  timestamp: string;     // ISO 8601
  agentId: string;
  model: string;
  tokensUsed: number;    // Extracted from usageMetadata
  durationSeconds: number;
  status: 'success' | 'error';
}
6. API Integration Strategy
The application acts as a direct client to the Google GenAI REST/gRPC infrastructure via the SDK.
6.1 Service Layer (geminiService.ts)
generateContent:
Used by Agent Runner and Note Keeper.
Pattern: Single-shot generation (ai.models.generateContent).
Configuration: Sets temperature: 0.2 to ensure deterministic, hallucination-resistant outputs suitable for regulatory compliance.
chatWithData:
Used by FDA Data Lab.
Pattern: Stateful chat session (ai.chats.create).
Maps internal ChatMessage array to Gemini SDK's Content parts structure.
6.2 Error Handling
Global try/catch blocks wrap API calls.
Failures (4xx/5xx errors, quota limits) are caught.
UI displays AlertCircle banners with error messages.
Failed runs are logged in the Dashboard with status: 'error' to track stability.
7. User Experience & Interface Guidelines
7.1 Layout Philosophy
Sidebar-Driven: Fixed left navigation ensures context switching is instant.
Glassmorphism: High usage of transparency (bg-white/10, backdrop-blur-md) allows the painter style gradients to bleed through UI elements, creating depth.
Responsive: Flexbox layouts ensure panels resize gracefully, though optimized for desktop viewing of documents.
7.2 Feedback Mechanisms
Loading States:
Spinners inside buttons (Run Agent).
Skeleton loaders or pulse animations (Processing... overlay in Notes).
Visual Cues:
Status API dot (Green/Red) in the header.
Document count indicators.
7.3 Accessibility (A11y)
Semantic HTML tags (nav, main, h1-h3).
Contrast ratios are maintained via dynamic Tailwind classes (e.g., adjusting text color based on the selected theme's brightness, though currently primarily optimized for dark/rich backgrounds).
Keyboard navigation support for form inputs.
8. Security & Compliance Considerations
8.1 Data Privacy
Zero-Persistence: The application does not use a backend database. All uploaded documents, chat history, and API keys exist only in the browser's React state.
Refresh-Wipe: Reloading the page clears all sensitive data, a desired feature for shared regulatory workstations to prevent accidental data leakage.
8.2 API Key Security
The application supports environment variable injection during build/deployment (process.env.API_KEY).
If a user inputs a key manually, it is held in memory only. It is not transmitted to any 3rd party besides Google's endpoints.
8.3 Regulatory Accuracy
Temperature Control: Agents are hardcoded with low temperature (0.2) to minimize creativity and maximize adherence to the provided source text.
Grounding: Prompts explicitly instruct agents to "Identify missing elements" and "Do not invent data," crucial for FDA 510(k) integrity.
9. Future Extensibility Roadmap
9.1 PDF Parsing Integration
Current State: Accepts text/markdown.
Specification: Integrate a client-side library (e.g., pdf.js) to extract text from PDF submission files automatically upon upload.
9.2 Export Capabilities
Current State: View output in UI.
Specification: Add "Download Report" functionality to bundle Agent outputs and Run Logs into a ZIP or PDF file for official archiving.
9.3 Custom Agent Builder
Current State: Agents are hardcoded in constants.ts.
Specification: Add a UI to allow users to define custom System Prompts and save them to localStorage as user-defined agents.
9.4 Multi-Modal Analysis
Current State: Text-only.
Specification: Update geminiService to accept Base64 image strings, enabling analysis of device diagrams, labels, and packaging photos within the Agent Runner.
10. Conclusion
The FDA 510(k) Agentic Review WOW Studio represents a paradigm shift in regulatory software. By combining rigid, compliance-oriented agent configurations with a fluid, artistically inspired user interface, it reduces the cognitive load on reviewers. The technical architecture—lean, client-side, and API-first—ensures rapid deployment and maximum data privacy, making it a robust tool for modern medical device regulatory workflows.
