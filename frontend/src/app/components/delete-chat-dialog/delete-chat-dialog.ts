import { Component, Inject } from '@angular/core';
import { MAT_DIALOG_DATA, MatDialogRef, MatDialogModule } from '@angular/material/dialog';
import { MatButtonModule } from '@angular/material/button';
import { CommonModule } from '@angular/common';

export interface DeleteChatDialogData {
  title: string;
}

@Component({
  selector: 'app-delete-chat-dialog',
  standalone: true,
  imports: [
    CommonModule,
    MatDialogModule,
    MatButtonModule
  ],
  templateUrl: './delete-chat-dialog.html',
  styleUrl: './delete-chat-dialog.css'
})
export class DeleteChatDialog {
  constructor(
    public dialogRef: MatDialogRef<DeleteChatDialog>,
    @Inject(MAT_DIALOG_DATA) public data: DeleteChatDialogData
  ) {}

  onCancel(): void {
    this.dialogRef.close(false); // Close dialog, indicating no deletion
  }

  onConfirmDelete(): void {
    this.dialogRef.close(true); // Close dialog, indicating deletion is confirmed
  }
}