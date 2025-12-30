import { Component, Inject } from '@angular/core';
import {
  MAT_DIALOG_DATA,
  MatDialogRef,
  MatDialogModule,
} from '@angular/material/dialog';
import { FormsModule } from '@angular/forms';
import { MatFormFieldModule } from '@angular/material/form-field';
import { MatInputModule } from '@angular/material/input';
import { MatButtonModule } from '@angular/material/button';

export interface RenameChatDialogData {
  currentTitle: string;
}

@Component({
  selector: 'app-rename-chat-dialog',
  standalone: true,
  imports: [
    MatDialogModule,
    FormsModule,
    MatFormFieldModule,
    MatInputModule,
    MatButtonModule,
  ],
  templateUrl: './rename-chat-dialog.html',
  styleUrl: './rename-chat-dialog.css',
})
export class RenameChatDialog {
  newTitle: string;

  constructor(
    public dialogRef: MatDialogRef<RenameChatDialog>,
    @Inject(MAT_DIALOG_DATA) public data: RenameChatDialogData
  ) {
    this.newTitle = data.currentTitle;
  }

  onCancel(): void {
    this.dialogRef.close();
  }

  onRename(): void {
    if (this.newTitle.trim()) {
      // Closes the dialog, returning the new title.
      this.dialogRef.close(this.newTitle.trim());
    }
  }
}
