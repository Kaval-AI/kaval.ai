import { Input, Component } from '@angular/core';
import { CommonModule } from '@angular/common';
import { JsonViewModule } from 'nxt-json-view';

class Segment {
  constructor(
    public type: 'text' | 'json',
    public content: string
  ) {}
}

@Component({
  selector: 'app-system-prompt',
  imports: [CommonModule, JsonViewModule],
  templateUrl: './system-prompt.html',
  styleUrl: './system-prompt.css',
})
export class SystemPrompt {
  @Input() text: string = '';

  getSegments(): Segment[] {
    return this.parseText(this.text);
  }

  private parseText(text: string): Segment[] {
    const segments: Segment[] = [];
    let lastIndex = 0; // The index in 'text' where the last segment ended.

    // State variables for the parser
    let openBraceCount = 0;
    let openBracketCount = 0;
    let inString = false;
    let escaped = false;
    let jsonStart = -1;
    let structuralChar: '{' | '[' | null = null; // Tracks which structure started the blob

    for (let i = 0; i < text.length; i++) {
      const char = text[i];

      // 1. Handle Escape sequence within a string
      if (inString && char === '\\') {
        escaped = !escaped; // Toggle (escaped state only applies to the next character)
        continue;
      }

      // 2. Handle String boundaries (outside of an escape sequence)
      if (char === '"' && !escaped) {
        inString = !inString;
      }

      // Reset escaped flag (it only protects the character immediately following the '\')
      if (escaped) {
        escaped = false;
      }

      // 3. Handle Structural Characters (only outside of strings)
      if (!inString) {
        // --- Start Detection ---
        if (char === '{' || char === '[') {
          if (openBraceCount === 0 && openBracketCount === 0) {
            // We found a potential start of a blob.

            // 3a. Capture the preceding plain text first.
            if (i > lastIndex) {
              segments.push(new Segment('text', text.substring(lastIndex, i)));
            }
            jsonStart = i;
            structuralChar = char;
            lastIndex = i; // Mark the beginning of the new potential segment
          }

          // Increment counters
          if (char === '{') {
            openBraceCount++;
          }
          if (char === '[') {
            openBracketCount++;
          }
        }
        // --- End Detection ---
        else if (char === '}' || char === ']') {
          // Decrement counters if they are positive
          if (char === '}' && openBraceCount > 0) {
            openBraceCount--;
          }
          if (char === ']' && openBracketCount > 0) {
            openBracketCount--;
          }

          // Check for balancing and successful extraction
          const isClosed = openBraceCount === 0 && openBracketCount === 0;
          const isMatchingStarter =
            (char === '}' && structuralChar === '{') ||
            (char === ']' && structuralChar === '[');

          if (jsonStart !== -1 && isClosed && isMatchingStarter) {
            // The blob has successfully closed
            const jsonCandidate = text.substring(jsonStart, i + 1);

            try {
              const parsed = JSON.parse(jsonCandidate);
              segments.push(new Segment('json', parsed));
            } catch (e) {
              // Failed JSON parsing (e.g., bad syntax inside the balanced structure) -> treat as text
              segments.push(new Segment('text', jsonCandidate));
            }

            // Reset state variables for the next iteration
            jsonStart = -1;
            structuralChar = null;
            lastIndex = i + 1; // Start next segment search immediately after the closing brace/bracket
          }
        }
      }
    }

    // 4. Final step: Add any remaining text
    if (lastIndex < text.length) {
      // This includes remaining plain text, or an unclosed blob treated as plain text
      segments.push(new Segment('text', text.substring(lastIndex)));
    }

    return segments;
  }
}
