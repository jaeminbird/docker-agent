package runtime

import (
	"context"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/docker/docker-agent/pkg/agent"
	"github.com/docker/docker-agent/pkg/chat"
	"github.com/docker/docker-agent/pkg/config/latest"
	"github.com/docker/docker-agent/pkg/environment"
	"github.com/docker/docker-agent/pkg/model/provider/openai"
	"github.com/docker/docker-agent/pkg/session"
	"github.com/docker/docker-agent/pkg/team"
	"github.com/docker/docker-agent/pkg/tools"
)

// TestRunStream_LengthDropsCutOffToolCalls drives a real *openai.Client
// through LocalRuntime: a turn that hits the token limit while streaming a
// tool call must end as length without running the truncated call, and the
// persisted assistant message must not keep a call without a result.
func TestRunStream_LengthDropsCutOffToolCalls(t *testing.T) {
	t.Parallel()

	sse := `data: {"id":"c1","object":"chat.completion.chunk","created":1,"model":"m","choices":[{"index":0,"delta":{"role":"assistant","tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"write","arguments":"{\"path\":\"notes.txt\",\"content\":\"hel"}}]},"finish_reason":null}]}` + "\n\n" +
		`data: {"id":"c1","object":"chat.completion.chunk","created":1,"model":"m","choices":[{"index":0,"delta":{},"finish_reason":"length"}],"usage":{"prompt_tokens":7,"completion_tokens":3,"total_tokens":10}}` + "\n\n" +
		"data: [DONE]\n\n"

	var requests atomic.Int64
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		requests.Add(1)
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = w.Write([]byte(sse))
	}))
	t.Cleanup(srv.Close)

	cfg := latest.ModelConfig{
		Provider:     "openai",
		Model:        "gpt-4o",
		BaseURL:      srv.URL,
		ProviderOpts: map[string]any{"api_type": "openai_chatcompletions"},
	}
	client, err := openai.NewClient(t.Context(), &cfg,
		environment.NewMapEnvProvider(map[string]string{"OPENAI_API_KEY": "test-key"}))
	require.NoError(t, err)

	var toolRuns atomic.Int64
	write := tools.Tool{
		Name:       "write",
		Parameters: map[string]any{"type": "object"},
		Handler: func(context.Context, tools.ToolCall, tools.Runtime) (*tools.ToolCallResult, error) {
			toolRuns.Add(1)
			return tools.ResultSuccess("ok"), nil
		},
	}
	root := agent.New("root", "test", agent.WithModel(client), agent.WithToolSets(newStubToolSet(nil, []tools.Tool{write}, nil)))
	rt, err := NewLocalRuntime(t.Context(), team.New(team.WithAgents(root)), WithSessionCompaction(false), WithModelStore(mockModelStore{}))
	require.NoError(t, err)

	sess := session.New(session.WithUserMessage("hi"), session.WithToolsApproved(true))
	sess.NonInteractive = true

	var stopped *StreamStoppedEvent
	var retracted []string
	for ev := range rt.RunStream(t.Context(), sess) {
		switch e := ev.(type) {
		case *ErrorEvent:
			t.Fatalf("unexpected error event: %s", e.Error)
		case *StreamStoppedEvent:
			stopped = e
		case *ToolCallsRetractedEvent:
			retracted = append(retracted, e.ToolCallIDs...)
		}
	}

	require.NotNil(t, stopped)
	assert.Equal(t, chat.FinishReasonLength, stopped.FinishReason)
	assert.Zero(t, toolRuns.Load(), "a cut-off tool call must not run")
	assert.Equal(t, int64(1), requests.Load())
	assert.Equal(t, []string{"call_1"}, retracted)
	for _, m := range sess.GetAllMessages() {
		assert.Empty(t, m.Message.ToolCalls, "no call may be persisted without a result")
	}
}
