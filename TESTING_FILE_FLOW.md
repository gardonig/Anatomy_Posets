# File Selection Flow Test Guide

## Overview
This guide documents the redesigned file selection flow in `src/anatomy_poset/gui/main_window.py` (lines 410–481). The key improvement is asking the user whether to continue from an existing file or start anew **before** opening a file dialog, eliminating confusing OS "replace file?" warnings.

## Architecture Changes

### Before
1. "Choose where to save this query's poset" → user picks a file
2. System asks "Replace existing file?" if path exists
3. If new: create matrices; if existing: load all three axes

**Problems:**
- OS warning seemed to imply overwriting, but code doesn't overwrite
- Timing of intent (new vs. continue) came after file selection
- User had to understand file dialog behavior to know what "replace" meant

### After
1. **First dialog:** "Start New or Continue?" (Yes/No)
   - Yes → getOpenFileName() to load existing
   - No → getSaveFileName() with DontConfirmOverwrite to create new
2. **No OS warnings** because:
   - getOpenFileName() has no replace semantics
   - getSaveFileName() suppresses confirmation with `QFileDialog.Option.DontConfirmOverwrite`

## Test Scenarios

### Scenario 1: Start New Session (No existing file)
**Steps:**
1. Launch app: `python run.py`
2. Load a structures file (e.g., `test_structures.json`)
3. Click "Start" button
4. Dialog appears: "Do you want to continue from an existing saved session or start a new one?"
5. Click "No"

**Expected:**
- "Choose where to save this new poset" dialog opens (file picker)
- **NO** OS "Replace file?" or confirmation dialog appears
- Navigate to `data/posets/` folder (or wherever)
- Enter a filename (e.g., `mysession.json`)
- Click Save
- App initializes three empty tri-valued matrices
- Query dialog appears with first question ready

**Verification:**
- ✓ No OS warning window appears
- ✓ File is created with correct structure (can check JSON afterward)
- ✓ Query dialog proceeds normally

---

### Scenario 2: Continue From Existing Session
**Setup:**
- An existing poset file exists (e.g., `data/posets/tests/test1.json`)

**Steps:**
1. Launch app: `python run.py`
2. Load same structures file that was used for the existing poset
3. Click "Start" button
4. Dialog appears: "Do you want to continue from an existing saved session or start a new one?"
5. Click "Yes"

**Expected:**
- "Select saved poset file to continue from" file picker opens
- No OS dialogs
- Navigate to the existing file (e.g., `data/posets/tests/test1.json`)
- Click Open
- App loads all three matrix axes from file
- Query dialog appears, showing the next unpaired structures

**Verification:**
- ✓ No OS warning/confirmation window
- ✓ Loaded matrices reflected in progress bar (e.g., partially filled if session was partially done)
- ✓ Query dialog picks up where file left off
- ✓ After answering a question and saving, **only the current axis matrix updates**, while the other two axes remain unchanged (verify in saved JSON)

---

### Scenario 3: Cancel File Selection
**Steps:**
1. Click "Start" button
2. Dialog: "Start New or Continue?"
3. Click "No" (or "Yes" - both should work the same way)
4. File picker opens
5. Click "Cancel" in the file picker

**Expected:**
- File picker closes
- App returns to main window
- "Start" button is re-enabled (user can try again)

**Verification:**
- ✓ No error dialogs
- ✓ App state unchanged (no file created/loaded)

---

### Scenario 4: Load Invalid File Gracefully
**Steps:**
1. Click "Start" button
2. Dialog: "Start New or Continue?"
3. Click "Yes"
4. Pick a file that is **not** a valid poset JSON (e.g., pick `test_structures.json` by mistake)
5. Click Open

**Expected:**
- Error dialog appears: "Failed to load file — Could not read the selected file. The session was not started."
- App returns to main window
- "Start" button is re-enabled

**Verification:**
- ✓ Graceful error handling
- ✓ No crash
- ✓ User can retry

---

### Scenario 5: Multi-Axis Preservation
**Steps:**
1. Create a new session for structures file
2. Choose vertical axis, answer several questions, save
3. Switch to mediolateral axis in the same session, answer several questions, save
4. Close and reopen the session (load the file)

**Expected:**
- Load dialog shows the file
- When querying vertical axis again, you pick up where you left off
- When you switch to mediolateral axis in the new session, those questions are fresh (null), but vertical matrix is still there

**Verification (by inspecting JSON):**
- ✓ `matrix_vertical` has some +1/-1/0 values
- ✓ `matrix_mediolateral` has null (None) and some values
- ✓ `matrix_anteroposterior` has all null

**Verification (in GUI):**
- ✓ Vertical axis load shows expected progress bar
- ✓ Mediolateral axis is fresh
- ✓ After answering in mediolateral and saving, rescan shows both matrices preserved

---

## Code Inspection Points

If manual testing is not possible, code inspection can verify correctness:

1. **Line 415–421:** QMessageBox condition check
   - ✓ Logic: `if reply == QMessageBox.StandardButton.Yes`

2. **Line 425–430:** getOpenFileName() parameters
   - ✓ No `options=` that would cause "replace" warning
   - ✓ Default filter set to JSON files

3. **Line 456–461:** getSaveFileName() parameters
   - ✓ Option `QFileDialog.Option.DontConfirmOverwrite` present
   - ✓ Prevents OS confirmation dialog

4. **Line 438–443:** Load path
   - ✓ All three matrices loaded: `matrix_vertical`, `matrix_mediolateral`, `matrix_anteroposterior`

5. **Line 475–481:** Save path
   - ✓ All three matrices saved: `matrix_vertical`, `matrix_mediolateral`, `matrix_anteroposterior`

---

## Summary

The redesigned flow eliminates the confusing "Replace file?" OS dialog by:
1. **Asking intent first** (continue vs. new) as a simple Yes/No
2. **Using correct file dialogs** for each intent:
   - getOpenFileName() for continuing (no replace semantics)
   - getSaveFileName(DontConfirmOverwrite) for new (suppresses confirmation)
3. **Ensuring clear semantics:** Continuing preserves all data; new starts fresh matrices

Expected outcome after testing: **No misleading OS dialogs, clear user intent, correct matrix loading/saving.**
