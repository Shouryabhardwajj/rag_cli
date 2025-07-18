## Setup

1.  **Clone the Repository:**
    ```bash
    git clone [https://github.com/Shouryabhardwajj/rag_cli.git](https://github.com/Shouryabhardwajj/rag_cli.git)
    cd rag_cli
    ```

2.  **Set up Gemini API Key:**
    Obtain a key from [Google AI Studio](https://aistudio.google.com/) and set it as an environment variable:

    * **Linux/macOS:**
        ```bash
        export GOOGLE_API_KEY="YOUR_GEMINI_API_KEY_HERE"
        # Add to ~/.bashrc, ~/.zshrc, or ~/.profile for persistence, then source the file.
        ```
    * **Windows (Command Prompt):**
        ```cmd
        set GOOGLE_API_KEY="YOUR_GEMINI_API_KEY_HERE"
        ```
    * **Windows (PowerShell):**
        ```powershell
        $env:GOOGLE_API_KEY="YOUR_GEMINI_API_KEY_HERE"
        ```

3.  **Set up Virtual Environment:**
    ```bash
    python -m venv venv
    ```

4.  **Activate Virtual Environment:**
    * **Linux/macOS:**
        ```bash
        source venv/bin/activate
        ```
    * **Windows (Command Prompt):**
        ```cmd
        .\venv\Scripts\activate
        ```
    * **Windows (PowerShell)::**
        ```powershell
        .\venv\Scripts\Activate.ps1
        ```

5.  **Install Requirements:**
    ```bash
    pip install -r requirements.txt
    ```

## Running the Project

With the virtual environment active and requirements installed:

```bash
python main.py 
