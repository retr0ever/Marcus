# Marcus User Guide

## Getting Started

1. **Launch the UI**
   ```bash
   cd marcus-codebase
   source .venv/bin/activate
   streamlit run marcus_ui/app.py
   ```

2. **Initialise the Pipeline**
   - Navigate to **Settings**
   - Click **Initialise Pipeline**
   - Wait for models to load

---

## Features

### Live Detection
Capture faces from your webcam and match against enrolled identities.

1. Go to **Live Detection**
2. Allow camera access
3. Click the capture button
4. View detected faces and matches

**Settings:**
- **Confidence Threshold**: Minimum detection confidence (0.3–0.95)
- **Similarity Threshold**: Minimum match similarity (0.3–0.95)
- **Top K**: Number of matches to display

---

### Photo Search
Upload a photograph to find matching identities.

1. Go to **Photo Search**
2. Upload an image (JPG, PNG, WEBP)
3. Review detected faces and matches

**Batch Mode:** Upload multiple images and click **Search All**.

---

### Enrol Identity
Add new identities to the database.

1. Go to **Enrol Identity**
2. Enter the person's name
3. Select data source (manual, public, consented, dataset)
4. Upload one or more photographs
5. Confirm consent checkbox
6. Click **Enrol Identity**

**View Enrolled:** Scroll down to see all enrolled identities. Use the search box to filter by name.

**Delete:** Click **Delete** next to any identity, then confirm.

---

### Settings
Configure and manage the system.

**Initialise Pipeline:**
- Default configuration or custom YAML file
- Manual configuration for detection, embedding, matching, and compliance settings

**Actions:**
- **Save Data**: Persist current state to disk
- **Reload Models**: Refresh loaded models
- **Reset Pipeline**: Clear and reinitialise

**Export/Import:** Save or load configuration as YAML.

---

## Tips

- Use well-lit, front-facing photographs
- Enrol multiple photographs per identity for better accuracy
- Lower similarity thresholds increase matches but reduce precision
- Check **System Information** in Settings for device capabilities

---

## Keyboard Shortcuts

| Action | Shortcut |
|--------|----------|
| Refresh page | Cmd + R |
| Hard refresh | Cmd + Shift + R |

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Pipeline not initialised | Go to Settings and click Initialise Pipeline |
| No faces detected | Ensure good lighting and clear visibility |
| Slow performance | Use GPU/MPS if available, or reduce image resolution |
| Camera not working | Check browser permissions |
