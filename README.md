## API Endpoints
---
### 1. Ask question/follow up questions
**Request:**

```
POST /chat
```

**Headers:**

```
Content-Type: application/json
```

**Body Example:**

```json
{
    "question": "What is this document about?",
    "session_id": "test-session-123"
}
```

- **`question`**: The query or message sent to the chatbot.
- **`session_id`**: Unique identifier for the chat session.

---
### 2. Clear chat history

**Request:**

```
DELETE /clear-history/{session_id}
```

**Path Parameter:**

- **`session_id`**: The ID of the session whose history is to be cleared.

**Headers:**

```
Accept: application/json
```

---

### 3. Reinit

**Request:**

```
POST /reinitialize
```

**Headers:**

```
Accept: application/json
```

