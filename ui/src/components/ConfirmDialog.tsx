import { useEffect, useRef } from "react";

export type ConfirmRequest = {
  title: string;
  body: string;
  confirmLabel: string;
  tone?: "danger" | "default";
  onConfirm: () => void;
};

type Props = {
  request: ConfirmRequest | null;
  onClose: () => void;
};

export function ConfirmDialog({ request, onClose }: Props) {
  const confirmRef = useRef<HTMLButtonElement>(null);
  const previousFocus = useRef<Element | null>(null);

  useEffect(() => {
    if (!request) return;
    previousFocus.current = document.activeElement;
    confirmRef.current?.focus();
    function onKeyDown(event: KeyboardEvent) {
      if (event.key === "Escape") onClose();
    }
    window.addEventListener("keydown", onKeyDown);
    return () => {
      window.removeEventListener("keydown", onKeyDown);
      if (previousFocus.current instanceof HTMLElement) {
        previousFocus.current.focus();
      }
    };
  }, [request, onClose]);

  if (!request) return null;

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div
        className="modal"
        role="alertdialog"
        aria-modal="true"
        aria-labelledby="confirm-title"
        aria-describedby="confirm-body"
        onClick={(event) => event.stopPropagation()}
      >
        <h2 id="confirm-title">{request.title}</h2>
        <p id="confirm-body">{request.body}</p>
        <div className="modal-actions">
          <button type="button" className="btn" onClick={onClose}>
            Cancel
          </button>
          <button
            ref={confirmRef}
            type="button"
            className={`btn ${request.tone === "danger" ? "danger" : "primary"}`}
            onClick={() => {
              request.onConfirm();
              onClose();
            }}
          >
            {request.confirmLabel}
          </button>
        </div>
      </div>
    </div>
  );
}
