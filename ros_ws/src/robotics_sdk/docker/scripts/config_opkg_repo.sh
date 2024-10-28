#!/bin/bash

#  Copyright (C) 2021 Texas Instruments Incorporated - http://www.ti.com/
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions
#  are met:
#
#    Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#
#    Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the
#    distribution.
#
#    Neither the name of Texas Instruments Incorporated nor the names of
#    its contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
#  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
#  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
#  A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
#  OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
#  SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
#  LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
#  DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
#  THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
#  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# Should run on PSDK host linux
OPKG_REPO_DIR=/opkg_repo

FILE=/etc/opkg/base-feeds.conf
if [ -f "${FILE}_org" ]
then
	echo "It looks like $FILE already updated. Here is content:"
	echo "cat $FILE"
	cat $FILE
else
	mv "$FILE" "${FILE}_org"
	# Update base-feeds.conf to use a local OPKG repository
	echo "# Original $FILE file was copied to ${FILE}_org" > $FILE
	echo "Updated $FILE"
	echo "cat $FILE"
	cat $FILE
fi

FILE=/etc/opkg/opkg.conf
if [ -f "${FILE}_org" ]
then
	echo ""
	echo "It looks like $FILE already updated. Here is content:"
	echo "cat $FILE"
	cat $FILE
else
	mv "$FILE" "${FILE}_org"
	# Update opkg.conf to use a local OPKG repository
	echo "# Original $FILE file was copied to ${FILE}_org" > $FILE
	echo "# Settings for a local OPKG repository" >> $FILE
	echo "src/gz local file://$OPKG_REPO_DIR"     >> $FILE
	echo "dest root /"                            >> $FILE
	echo "option lists_dir /var/lib/opkg/lists"   >> $FILE

	echo ""
	echo "Updated $FILE"
	echo "cat $FILE"
	cat $FILE
fi

echo ""
echo "$0: Completed"
